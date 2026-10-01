#!/usr/bin/env python3
"""Gluino / jet validation plots -- three live read backends.

  * "uproot"  flat TTree (NanoAOD-style GenPart_* arrays), chunked + vectorized.
              Needs: pip install uproot awkward numpy.  No CMSSW, no CVMFS.
  * "pyroot"  the same flat TTree via bare PyROOT GetEntry(). Fallback when
              uproot is absent. Slower, but zero extra dependencies.
  * "fwlite"  EDM AOD/MiniAOD, vector<reco::GenParticle> + vector<reco::GenJet>.
              Requires cmsenv.
"""

import argparse
import itertools
import math
import os
import sys

import ROOT

try:
    import uproot
    import awkward as ak
    import numpy as np
    HAVE_UPROOT = True
except ImportError:
    uproot = ak = np = None
    HAVE_UPROOT = False

try:
    from DataFormats.FWLite import Events, Handle
    HAVE_FWLITE = True
except ImportError:
    Events = Handle = None
    HAVE_FWLITE = False

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(1111)

GLUINO_PDGID = 1000021

GENPART_FIELDS = ("pt", "eta", "phi", "mass",
                  "pdgId", "status", "statusFlags", "genPartIdxMother")


# ---------------------------------------------------------------------------
# reco::GenStatusFlags bit order
# ---------------------------------------------------------------------------

STATUS_FLAG_BITS = {
    "isPrompt":                           0,
    "isDecayedLeptonHadron":              1,
    "isTauDecayProduct":                  2,
    "isPromptTauDecayProduct":            3,
    "isDirectTauDecayProduct":            4,
    "isDirectPromptTauDecayProduct":      5,
    "isDirectHadronDecayProduct":         6,
    "isHardProcess":                      7,
    "fromHardProcess":                    8,
    "isHardProcessTauDecayProduct":       9,
    "isDirectHardProcessTauDecayProduct": 10,
    "fromHardProcessBeforeFSR":           11,
    "isFirstCopy":                        12,
    "isLastCopy":                         13,
    "isLastCopyBeforeFSR":                14,
}

FLAG_NAMES_ORDERED = sorted(STATUS_FLAG_BITS, key=STATUS_FLAG_BITS.get)

BIT_IS_LAST_COPY            = STATUS_FLAG_BITS["isLastCopy"]
BIT_IS_LAST_COPY_BEFORE_FSR = STATUS_FLAG_BITS["isLastCopyBeforeFSR"]
BIT_IS_HARD_PROCESS         = STATUS_FLAG_BITS["isHardProcess"]

SELECTOR_BITS = {
    "lastcopy":           BIT_IS_LAST_COPY,
    "lastcopy_beforefsr": BIT_IS_LAST_COPY_BEFORE_FSR,
    "hardprocess":        BIT_IS_HARD_PROCESS,
}

SELECTOR_METHODS = {
    "lastcopy":           "isLastCopy",
    "lastcopy_beforefsr": "isLastCopyBeforeFSR",
    "hardprocess":        "isHardProcess",
}


class FlatStatusFlags(object):
    """Bitmask wearing the reco::GenStatusFlags method interface."""

    __slots__ = ("bits",)

    def __init__(self, bits):
        self.bits = int(bits)

    def test(self, name):
        return bool((self.bits >> STATUS_FLAG_BITS[name]) & 1)

    def flags_set(self):
        return [n for n in FLAG_NAMES_ORDERED if self.test(n)]

    def __repr__(self):
        return "FlatStatusFlags(%d: %s)" % (self.bits, ",".join(self.flags_set()))


def _install_flag_methods():
    for _name in STATUS_FLAG_BITS:
        def _make(n):
            return lambda self: self.test(n)
        setattr(FlatStatusFlags, _name, _make(_name))


_install_flag_methods()


# ---------------------------------------------------------------------------
# flat-array shims
# ---------------------------------------------------------------------------

class FlatJet(object):
    """Minimal jet quacking like reco::GenJet for the kinematic accessors."""

    __slots__ = ("_pt", "_eta", "_phi", "_mass")

    def __init__(self, pt, eta, phi, mass):
        self._pt, self._eta = float(pt), float(eta)
        self._phi, self._mass = float(phi), float(mass)

    def pt(self):   return self._pt
    def eta(self):  return self._eta
    def phi(self):  return self._phi
    def mass(self): return self._mass

    def energy(self):
        p = self._pt * math.cosh(self._eta)
        return math.sqrt(p * p + self._mass * self._mass)


class FlatGenParticle(object):
    """One GenPart entry quacking like a reco::GenParticle."""

    __slots__ = ("ev", "idx")

    def __init__(self, event, idx):
        self.ev = event
        self.idx = int(idx)

    def pt(self):   return float(self.ev.pt[self.idx])
    def eta(self):  return float(self.ev.eta[self.idx])
    def phi(self):  return float(self.ev.phi[self.idx])
    def mass(self): return float(self.ev.mass[self.idx])

    def energy(self):
        p = self.pt() * math.cosh(self.eta())
        m = self.mass()
        return math.sqrt(p * p + m * m)

    def p4(self):
        return ROOT.Math.PtEtaPhiMVector(self.pt(), self.eta(),
                                         self.phi(), self.mass())

    def pdgId(self):  return int(self.ev.pdgId[self.idx])
    def status(self): return int(self.ev.status[self.idx])

    def statusFlags(self):
        return self.ev.flags_obj(self.idx)

    def flag_bit(self, bit):
        return bool((int(self.ev.statusFlags[self.idx]) >> bit) & 1)

    def numberOfDaughters(self):
        return len(self.ev.children_of(self.idx))

    def daughter(self, i):
        return FlatGenParticle(self.ev, self.ev.children_of(self.idx)[i])

    def numberOfMothers(self):
        return 0 if int(self.ev.mother[self.idx]) < 0 else 1

    def mother(self, i=0):
        m = int(self.ev.mother[self.idx])
        return None if m < 0 else FlatGenParticle(self.ev, m)

    def key(self):
        return self.idx

    def __repr__(self):
        return ("FlatGenParticle(idx=%d pdgId=%d pt=%.1f status=%d)"
                % (self.idx, self.pdgId(), self.pt(), self.status()))


class FlatGenEvent(object):
    """Per-event view onto one slice of a chunk (numpy views, zero copy)."""

    __slots__ = ("pt", "eta", "phi", "mass", "pdgId", "status",
                 "statusFlags", "mother", "n", "_children", "_flags_cache")

    def __init__(self, pt, eta, phi, mass, pdgId, status, statusFlags, mother):
        self.pt, self.eta, self.phi, self.mass = pt, eta, phi, mass
        self.pdgId, self.status = pdgId, status
        self.statusFlags, self.mother = statusFlags, mother
        self.n = len(pt)
        self._children = None
        self._flags_cache = {}

    def _build_children(self):
        """Invert mother -> children.

        Self-mothers (genPartIdxMother[i] == i) are DROPPED. Official NanoAOD
        guarantees mother index < own index, but a custom flattener can emit a
        self-reference, which sends the FSR-aware walk in find_last_copy into
        an infinite loop at 100% CPU -- the previous version "runs forever" bug.
        """
        kids = [[] for _ in range(self.n)]
        for i in range(self.n):
            m = int(self.mother[i])
            if 0 <= m < self.n and m != i:
                kids[m].append(i)
        self._children = kids

    def children_of(self, idx):
        if self._children is None:
            self._build_children()
        return self._children[idx]

    def flags_obj(self, idx):
        f = self._flags_cache.get(idx)
        if f is None:
            f = FlatStatusFlags(int(self.statusFlags[idx]))
            self._flags_cache[idx] = f
        return f

    def particle(self, idx):
        return FlatGenParticle(self, idx)

    def particles(self):
        return [FlatGenParticle(self, i) for i in range(self.n)]


# ---------------------------------------------------------------------------
# shared physics helpers -- identical for every backend
# ---------------------------------------------------------------------------

MAX_WALK = 500

def has_status_flags(gp):
    try:
        gp.statusFlags()
        return True
    except Exception:
        return False


def _particle_key(gp):
    """Stable identity for cycle detection across both object models."""
    try:
        return ("flat", gp.idx)
    except AttributeError:
        pass
    try:
        return ("fw", gp.key().key())
    except Exception:
        return ("fw", id(gp))


def find_last_copy(gp, use_flags=True):
    """Walk self-copies down to the particle that actually decays.

    Flag path: trust reco::GenStatusFlags::isLastCopy(), which is what the
    generator itself recorded.

    Fallback path: walk same-pdgId daughters. The previous version required
    len(daughters) == 1, which breaks on FSR -- a gluino radiating a gluon is
    recorded as g~ -> g~ g, two daughters -- so the walk stopped early and the
    radiated gluon was mis-booked as a decay product, pushing the daughter
    invariant mass above the gluino pole mass. That condition is gone.
    """
    current = gp
    seen = {_particle_key(current)}

    for _ in range(MAX_WALK):
        if use_flags:
            try:
                if current.statusFlags().isLastCopy():
                    return current
            except Exception:
                use_flags = False

        ndau = current.numberOfDaughters()
        if ndau == 0:
            return current

        nxt = None
        for i in range(ndau):
            d = current.daughter(i)
            if d is not None and d.pdgId() == current.pdgId():
                nxt = d
                break
        if nxt is None:
            return current

        k = _particle_key(nxt)
        if k in seen:          # cycle in the mother/daughter links
            return current
        seen.add(k)
        current = nxt

    return current


def get_decay_daughters(gp, use_flags=True):
    last = find_last_copy(gp, use_flags)
    return [last.daughter(i) for i in range(last.numberOfDaughters())]


def is_susy_pdgid(pdgid):
    """SUSY lives in 1000001-1000039 and 2000001-2000015."""
    a = abs(pdgid)
    return (1000001 <= a <= 1000039) or (2000001 <= a <= 2000015)


def resolve_decay_chain(gp, use_flags=True, max_depth=40):
    """(intermediates, final_daughters) -- recurse through the SUSY cascade."""
    intermediates = []
    final_daughters = []
    visited = set()

    def _recurse(particle, depth):
        if depth > max_depth:
            return
        last = find_last_copy(particle, use_flags)
        k = _particle_key(last)
        if k in visited:
            return
        visited.add(k)

        ndau = last.numberOfDaughters()
        if ndau == 0:
            final_daughters.append(last)
            return

        for i in range(ndau):
            d = last.daughter(i)
            if d is None:
                continue
            if d.pdgId() == last.pdgId():
                continue          # residual self-copy left behind by pruning
            if is_susy_pdgid(d.pdgId()):
                intermediates.append(d)
                _recurse(d, depth + 1)
            else:
                final_daughters.append(d)

    _recurse(gp, 0)
    return intermediates, final_daughters


def invariant_mass(particles):
    """Sum four-momenta in plain Python.

    Cheaper than building a PtEtaPhiMVector per particle: the 3+3 jet pairing
    touches 10 combinations x 6 jets per event, and the object churn was a
    measurable slice of runtime once reading got fast.
    """
    if not particles:
        return 0.0
    e = px = py = pz = 0.0
    for p in particles:
        pt, eta, phi, m = p.pt(), p.eta(), p.phi(), p.mass()
        px += pt * math.cos(phi)
        py += pt * math.sin(phi)
        pz += pt * math.sinh(eta)
        mom = pt * math.cosh(eta)
        e += math.sqrt(mom * mom + m * m)
    m2 = e * e - (px * px + py * py + pz * pz)
    return math.sqrt(m2) if m2 > 0.0 else 0.0


def passes_selector(gp, selector, status, use_flags=True):
    if selector == "status":
        return gp.status() == status
    if not use_flags:
        return True
    try:
        return bool(getattr(gp.statusFlags(), SELECTOR_METHODS[selector])())
    except Exception:
        return True


def print_all_getters(obj):
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
            val = attr() if callable(attr) else attr
            print("%-30s %s" % (name, val))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# one event contract for every backend
# ---------------------------------------------------------------------------

class EventView(object):
    """What the analysis loop sees, regardless of file format."""

    __slots__ = ("index", "jets", "gluinos", "n_gluino", "n_gluino_sx",
                 "genparticles", "has_jets")

    def __init__(self, index, jets, gluinos, n_gluino, n_gluino_sx,
                 genparticles=None, has_jets=True):
        self.index = index
        self.jets = jets
        self.gluinos = gluinos
        self.n_gluino = n_gluino
        self.n_gluino_sx = n_gluino_sx
        self.genparticles = genparticles
        self.has_jets = has_jets


# ---------------------------------------------------------------------------
# backend detection
# ---------------------------------------------------------------------------

def list_branches(path):
    """Branch names of the Events tree, without touching dictionaries."""
    if HAVE_UPROOT:
        try:
            with uproot.open(path) as f:
                tree = f.get("Events")
                if tree is not None:
                    return list(tree.keys())
        except Exception:
            pass
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        return []
    tree = f.Get("Events")
    names = [b.GetName() for b in tree.GetListOfBranches()] if tree else []
    f.Close()
    return names


def detect_backend(path, requested, quiet=False):
    """auto -> uproot / pyroot / fwlite from the branch names."""
    if requested != "auto":
        return requested

    names = list_branches(path)
    flat = any(n == "nGenPart" or n.startswith("GenPart_") for n in names)
    edm = any("reco" in n and "_" in n and n.endswith(".") for n in names)

    if flat:
        chosen = "uproot" if HAVE_UPROOT else "pyroot"
        if not HAVE_UPROOT and not quiet:
            print("[info] flat TTree detected but uproot is missing -> slow "
                  "PyROOT reader. pip install uproot awkward numpy")
    elif edm or not names:
        chosen = "fwlite"
    else:
        chosen = "uproot" if HAVE_UPROOT else "pyroot"

    if not quiet:
        print("[info] backend auto-detected: %s" % chosen)
    return chosen


def find_jet_prefix(names, requested):
    """Pick a flat jet-array prefix; None if the file has no jets."""
    if requested:
        return requested if ("%s_pt" % requested) in names else None
    for cand in ("GenJet", "Jet"):
        if ("%s_pt" % cand) in names:
            return cand
    return None


# ---------------------------------------------------------------------------
# backend 1: uproot, chunked + vectorized
# ---------------------------------------------------------------------------

def iter_events_uproot(args):
    tree = uproot.open(args.infile)["Events"]
    names = set(tree.keys())

    missing = [f for f in GENPART_FIELDS if ("GenPart_%s" % f) not in names]
    if missing:
        raise SystemExit("missing GenPart branches: %s"
                         % ", ".join("GenPart_%s" % m for m in missing))

    jet_prefix = find_jet_prefix(names, args.jet_prefix)
    if jet_prefix is None:
        print("[info] no jet arrays in this file -> jet/HT/m3j plots skipped.")
    jet_fields = ("pt", "eta", "phi", "mass")

    branches = ["GenPart_%s" % f for f in GENPART_FIELDS]
    if jet_prefix:
        branches += ["%s_%s" % (jet_prefix, f) for f in jet_fields]
    if args.check_counts and "nGenPart" in names:
        branches.append("nGenPart")

    print("Total events in file: %d" % tree.num_entries)
    n_seen = 0
    done = False

    for chunk in tree.iterate(branches, step_size=args.step_size, library="ak"):
        if done:
            break

        pdg = chunk["GenPart_pdgId"]
        flags = chunk["GenPart_statusFlags"]
        ptj = chunk["GenPart_pt"]
        etaj = chunk["GenPart_eta"]

        # whole-chunk gluino mask -- replaces the per-event Python pre-scan
        mask = (pdg == GLUINO_PDGID) & (ptj > args.gluino_pt_min) \
            & (abs(etaj) < args.jet_eta_max)
        if args.selector == "status":
            sel_mask = mask & (chunk["GenPart_status"] == args.status)
        else:
            bit = SELECTOR_BITS[args.selector]
            sel_mask = mask & (((flags >> bit) & 1) == 1)

        counts_all = ak.to_numpy(ak.sum(mask, axis=1))
        counts_sel = ak.to_numpy(ak.sum(sel_mask, axis=1))
        keep = counts_all > 0

        nper = ak.to_numpy(ak.num(ptj))
        offs = np.concatenate([[0], np.cumsum(nper)]).astype(np.int64)

        if args.check_counts and "nGenPart" in chunk.fields:
            declared = ak.to_numpy(chunk["nGenPart"])
            bad = int(np.sum(declared != nper))
            if bad:
                print("[warn] nGenPart disagrees with array length in %d "
                      "event(s) of this chunk" % bad)

        # one flatten per branch per chunk; per-event slices are numpy views
        cols = {f: ak.to_numpy(ak.flatten(chunk["GenPart_%s" % f]))
                for f in GENPART_FIELDS}

        if jet_prefix:
            jnper = ak.to_numpy(ak.num(chunk["%s_pt" % jet_prefix]))
            joffs = np.concatenate([[0], np.cumsum(jnper)]).astype(np.int64)
            jcols = {f: ak.to_numpy(ak.flatten(chunk["%s_%s" % (jet_prefix, f)]))
                     for f in jet_fields}

        sel_idx_chunk = ak.to_numpy(
            ak.flatten(ak.local_index(ptj)[sel_mask])) if ak.any(sel_mask) else np.array([], dtype=np.int64)
        sel_counts = ak.to_numpy(ak.sum(sel_mask, axis=1))
        sel_offs = np.concatenate([[0], np.cumsum(sel_counts)]).astype(np.int64)

        for i in range(len(nper)):
            if args.max_events > 0 and n_seen >= args.max_events:
                done = True
                break
            n_seen += 1

            jets = []
            if jet_prefix:
                a, b = joffs[i], joffs[i + 1]
                jets = [FlatJet(jcols["pt"][a + k], jcols["eta"][a + k],
                                jcols["phi"][a + k], jcols["mass"][a + k])
                        for k in range(b - a)]

            if not keep[i] and not jets:
                continue

            gluinos = []
            if keep[i]:
                a, b = offs[i], offs[i + 1]
                fev = FlatGenEvent(*(cols[f][a:b] for f in GENPART_FIELDS))
                for k in range(sel_offs[i], sel_offs[i + 1]):
                    gluinos.append(FlatGenParticle(fev, sel_idx_chunk[k]))

            yield EventView(n_seen - 1, jets, gluinos,
                            int(counts_all[i]), int(counts_sel[i]),
                            has_jets=bool(jet_prefix))


# ---------------------------------------------------------------------------
# backend 2: flat TTree via PyROOT GetEntry  (restored from v4)
# ---------------------------------------------------------------------------

def iter_events_pyroot(args):
    f = ROOT.TFile.Open(args.infile)
    if not f or f.IsZombie():
        raise SystemExit("could not open %s" % args.infile)
    tree = f.Get("Events")
    if not tree:
        raise SystemExit("no 'Events' TTree in %s" % args.infile)

    names = [b.GetName() for b in tree.GetListOfBranches()]
    jet_prefix = find_jet_prefix(set(names), args.jet_prefix)
    if jet_prefix is None:
        print("[info] no jet arrays in this file -> jet/HT/m3j plots skipped.")
    jet_fields = ("pt", "eta", "phi", "mass")

    n_entries = tree.GetEntries()
    print("Total events in file: %d" % n_entries)

    limit = n_entries if args.max_events < 0 else min(n_entries, args.max_events)

    for i in range(limit):
        tree.GetEntry(i)
        n = int(getattr(tree, "nGenPart"))

        # Hoist the getattr OUT of the comprehension. The slow version did
        #     g = lambda fld: getattr(tree, "GenPart_%s" % fld)
        #     self.pt = [g("pt")[j] for j in range(n)]
        # which is one PyROOT proxy lookup per *particle* -- 8*nGenPart per
        # event. With nGenPart ~ 10k that is ~80k lookups/event.
        pdg_buf = getattr(tree, "GenPart_pdgId")
        pt_buf = getattr(tree, "GenPart_pt")
        eta_buf = getattr(tree, "GenPart_eta")

        # pre-scan: most events have no gluino, so bail before building
        # anything at all
        hits = [k for k in range(n)
                if int(pdg_buf[k]) == GLUINO_PDGID
                and float(pt_buf[k]) > args.gluino_pt_min
                and abs(float(eta_buf[k])) < args.jet_eta_max]

        jets = []
        if jet_prefix:
            njet = int(getattr(tree, "n%s" % jet_prefix)) \
                if ("n%s" % jet_prefix) in names else \
                len(getattr(tree, "%s_pt" % jet_prefix))
            jb = {fl: getattr(tree, "%s_%s" % (jet_prefix, fl)) for fl in jet_fields}
            jets = [FlatJet(jb["pt"][k], jb["eta"][k], jb["phi"][k], jb["mass"][k])
                    for k in range(njet)]

        if not hits and not jets:
            continue

        gluinos = []
        n_sel = 0
        if hits:
            # list(buf[:n]) copies at C level. The copy is deliberate: PyROOT
            # hands back the branch's internal buffer, which GetEntry()
            # overwrites in place, so holding the reference across events
            # silently yields the NEXT event's values.
            cols = {}
            for fld in GENPART_FIELDS:
                cols[fld] = list(getattr(tree, "GenPart_%s" % fld)[:n])
            fev = FlatGenEvent(*(cols[fld] for fld in GENPART_FIELDS))

            for k in hits:
                gp = FlatGenParticle(fev, k)
                if passes_selector(gp, args.selector, args.status,
                                   not args.no_status_flags):
                    gluinos.append(gp)
            n_sel = len(gluinos)

        yield EventView(i, jets, gluinos, len(hits), n_sel,
                        has_jets=bool(jet_prefix))

    f.Close()


# ---------------------------------------------------------------------------
# backend 3: FWLite EDM  (this is what v5 dropped)
# ---------------------------------------------------------------------------

def _require_fwlite():
    if not HAVE_FWLITE:
        raise SystemExit(
            "FWLite not found -- the 'fwlite' backend needs CMSSW.\n"
            "  source /cvmfs/cms.cern.ch/cmsset_default.sh\n"
            "  cd $CMSSW_BASE/src && cmsenv\n"
            "then re-run. For a flat NanoAOD-style TTree use --backend uproot "
            "instead; that path needs no CMSSW at all.")


def iter_events_fwlite(args):
    _require_fwlite()

    jetHandle = Handle("vector<%s>" % args.type)
    genHandle = Handle("vector<reco::GenParticle>")
    events = Events(args.infile)
    print("Total events in file: %d" % events.size())

    use_flags = not args.no_status_flags
    probed = False

    for i, event in enumerate(events):
        if args.max_events > 0 and i >= args.max_events:
            break

        jets = []
        event.getByLabel(args.label, jetHandle)
        if jetHandle.isValid():
            jets = [j for j in jetHandle.product()]

        gluinos = []
        n_all = n_sel = 0
        event.getByLabel(args.genparticle_label, genHandle)
        if genHandle.isValid():
            gps = genHandle.product()

            if use_flags and not probed:
                probed = True
                if len(gps) and not has_status_flags(gps[0]):
                    print("[warn] statusFlags() unavailable on this collection "
                          "-> falling back to tree walking.")
                    use_flags = False

            cands = [gp for gp in gps
                     if gp.pdgId() == GLUINO_PDGID
                     and gp.pt() > args.gluino_pt_min
                     and abs(gp.eta()) < args.jet_eta_max]
            n_all = len(cands)
            gluinos = [gp for gp in cands
                       if passes_selector(gp, args.selector, args.status, use_flags)]
            n_sel = len(gluinos)

        yield EventView(i, jets, gluinos, n_all, n_sel,
                        genparticles=None, has_jets=True)


def iter_events(args, backend):
    if backend == "uproot":
        if not HAVE_UPROOT:
            raise SystemExit("uproot not installed. pip install uproot awkward "
                             "numpy, or use --backend pyroot / fwlite.")
        return iter_events_uproot(args)
    if backend == "pyroot":
        return iter_events_pyroot(args)
    if backend == "fwlite":
        return iter_events_fwlite(args)
    raise SystemExit("unknown backend %r" % backend)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Gluino / jet validation plots (flat TTree or EDM)")
    p.add_argument("infile", help="Path to the input ROOT file")

    p.add_argument("--backend", default="auto",
                   choices=["auto", "uproot", "pyroot", "fwlite"],
                   help="Read path. 'auto' inspects the branch names.")
    p.add_argument("--branches", action="store_true",
                   help="Print branch names of the 'Events' tree and exit.")
    p.add_argument("--dump", action="store_true",
                   help="Print every method/value of the first jet and exit.")
    p.add_argument("--dump-gluino-decay", action="store_true",
                   help="Print gluino decay chains and exit (no plots).")

    p.add_argument("--label", default="ak4GenJets",
                   help="EDM jet collection module label (fwlite only)")
    p.add_argument("--type", default="reco::GenJet",
                   choices=["reco::GenJet", "pat::Jet", "reco::PFJet"],
                   help="C++ type of the EDM jet collection (fwlite only)")
    p.add_argument("--genparticle-label", default="genParticles",
                   help="EDM GenParticle module label (fwlite only)")
    p.add_argument("--jet-prefix", default=None,
                   help="Flat jet array prefix, e.g. GenJet or Jet. "
                        "Auto-detected; jets are skipped if absent.")

    p.add_argument("--selector", default="lastcopy",
                   choices=["lastcopy", "lastcopy_beforefsr",
                            "hardprocess", "status"],
                   help="Gluino selection. 'lastcopy' = isLastCopy(), the "
                        "gluino that actually decays (use for mass from "
                        "daughters). 'hardprocess' = production level.")
    p.add_argument("--status", type=int, default=22,
                   help="Required status when --selector status (default 22)")
    p.add_argument("--no-status-flags", action="store_true",
                   help="Ignore statusFlags and use tree walking instead.")

    p.add_argument("--max-events", type=int, default=-1,
                   help="Stop after N events (-1 = all)")
    p.add_argument("--step-size", default="100 MB",
                   help="uproot chunk size. A STRING is bytes; a bare int is "
                        "a number of events (default '100 MB').")
    p.add_argument("--check-counts", action="store_true",
                   help="Cross-check nGenPart against the array lengths.")

    p.add_argument("--jet-pt-min", type=float, default=20.0)
    p.add_argument("--jet-eta-max", type=float, default=2.4)
    p.add_argument("--gluino-pt-min", type=float, default=20.0)

    p.add_argument("--out", default="sample_validation_plots.pdf",
                   help="Output plot filename")
    p.add_argument("--root-out", default=None,
                   help="Histogram .root file (default: --out with .root)")
    p.add_argument("--plots-per-canvas", type=int, default=1,
                   help="Histograms per canvas/page (default 1)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# histograms
# ---------------------------------------------------------------------------

def book_histograms(with_jets):
    h = {}
    if with_jets:
        h.update({
            "njets":  ROOT.TH1F("h_njets", "Jets per event;n_{jets};Events", 20, 0, 20),
            "pt":     ROOT.TH1F("h_pt", "Jet p_{T};p_{T} [GeV];Jets", 100, 0, 1500),
            "eta":    ROOT.TH1F("h_eta", "Jet #eta;#eta;Jets", 100, -5, 5),
            "phi":    ROOT.TH1F("h_phi", "Jet #phi;#phi [rad];Jets", 100, -3.2, 3.2),
            "mass":   ROOT.TH1F("h_mass", "Jet mass;mass [GeV];Jets", 100, 0, 100),
            "energy": ROOT.TH1F("h_energy", "Jet energy;E [GeV];Jets", 100, 0, 1000),
            "nconst": ROOT.TH1F("h_nconst", "Jet n constituents;nConstituents;Jets", 100, 0, 100),
            "ndaugh": ROOT.TH1F("h_ndaugh", "Jet n daughters;nDaughters;Jets", 100, 0, 100),
            "ht":     ROOT.TH1F("h_ht", "Jet H_{T};H_{T} [GeV];Events", 100, 0, 4000),
            "m3j":    ROOT.TH1F("h_m3j", "3-jet system mass;M_{3j} [GeV];Triplets", 100, 0, 3000),
            "m3j_diff": ROOT.TH1F("h_m3j_diff", "Best 3-jet mass difference;|#DeltaM_{3j}| [GeV];Events", 100, 0, 1000),
        })
    h.update({
        "ngluino":    ROOT.TH1F("h_ngluino", "Gluinos per event;n_{#tilde{g}};Events", 30, 0, 30),
        "ngluino_sx": ROOT.TH1F("h_ngluino_sx", "Selected gluinos per event;n_{#tilde{g}};Events", 10, 0, 10),
        "gluino_mass": ROOT.TH1F("h_gluino_mass", "Gluino mass;mass [GeV];Gluinos", 100, 0, 3000),
        "gluino_pt":   ROOT.TH1F("h_gluino_pt", "Gluino p_{T};p_{T} [GeV];Gluinos", 100, 0, 2000),
        "gluino_ndaughters": ROOT.TH1F("h_gluino_ndaughters", "Gluino decay daughters;n_{daughters};Gluinos", 10, 0, 10),
        "gluino_dau_mass": ROOT.TH1F("h_gluino_dau_mass", "Invariant mass of gluino daughters;M_{daughters} [GeV];Gluinos", 100, 0, 3000),
        "dau_pdgid":   ROOT.TH1F("h_dau_pdgid", "Gluino daughter pdgId;pdgId;Daughters", 21, -10.5, 10.5),
        "ngluino_vs_status": ROOT.TH2F("h_ngluino_vs_status",
                                       "status vs n_{#tilde{g}};n_{#tilde{g}};status",
                                       30, 0, 30, 100, 0, 100),
        "flagbits": ROOT.TH1F("h_flagbits", "statusFlags bits set on selected gluinos;flag;Gluinos",
                              len(FLAG_NAMES_ORDERED), 0, len(FLAG_NAMES_ORDERED)),
    })
    for i, name in enumerate(FLAG_NAMES_ORDERED, start=1):
        h["flagbits"].GetXaxis().SetBinLabel(i, name)
    for hist in h.values():
        hist.SetDirectory(0)
    h["ngluino_vs_status"].SetStats(0)
    h["flagbits"].SetStats(0)
    return h


# ---------------------------------------------------------------------------
# inspect-and-exit modes
# ---------------------------------------------------------------------------

def mode_branches(args):
    names = list_branches(args.infile)
    if not names:
        print("No 'Events' tree, or file unreadable: %s" % args.infile)
        return
    print("Found %d branches in 'Events':\n" % len(names))
    for n in sorted(names):
        print(n)


def mode_dump(args, backend):
    for ev in iter_events(args, backend):
        if ev.jets:
            print_all_getters(ev.jets[0])
            return
    print("No jets found -- nothing to dump.")


def mode_dump_decay(args, backend):
    use_flags = not args.no_status_flags
    n_dumped = 0

    for ev in iter_events(args, backend):
        if not ev.gluinos:
            continue

        print("Event %d -> %d selected gluino(s) [selector=%s]:\n"
              % (ev.index, len(ev.gluinos), args.selector))
        for idx, gp in enumerate(ev.gluinos):
            print("  Gluino #%d: pt=%.1f eta=%.2f phi=%.2f mass=%.2f status=%d"
                  % (idx, gp.pt(), gp.eta(), gp.phi(), gp.mass(), gp.status()))
            try:
                print("    flags: %s"
                      % ",".join(n for n in FLAG_NAMES_ORDERED
                                 if getattr(gp.statusFlags(), n)()))
            except Exception:
                pass

            last = find_last_copy(gp, use_flags)
            if last is not gp and last.idx != getattr(gp, "idx", None):
                print("    -> last copy: pt=%.1f status=%d"
                      % (last.pt(), last.status()))

            intermediates, daughters = resolve_decay_chain(gp, use_flags)
            if intermediates:
                print("    -> %d intermediate SUSY particle(s):" % len(intermediates))
                for sp in intermediates:
                    print("         pdgId=%-8d status=%-4d pt=%-8.2f eta=%-6.2f mass=%.3f"
                          % (sp.pdgId(), sp.status(), sp.pt(), sp.eta(), sp.mass()))
            else:
                print("    -> no intermediate SUSY particles")

            print("    -> %d final daughter(s):" % len(daughters))
            for d in daughters:
                print("         pdgId=%-6d status=%-4d pt=%-8.2f eta=%-6.2f mass=%.3f"
                      % (d.pdgId(), d.status(), d.pt(), d.eta(), d.mass()))
            if daughters:
                print("    -> invariant mass of daughters: %.2f GeV\n"
                      % invariant_mass(daughters))

        n_dumped += 1
        # v2 fix: the old guard was `if iter_events >= args.max_events`, which
        # fires immediately under the --max-events -1 default (0 >= -1).
        if args.max_events > 0 and n_dumped >= args.max_events:
            return

    if n_dumped == 0:
        print("No event with a selected gluino found -- nothing to dump.")


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def draw_all(hists, args, order):
    order = [k for k in order if k in hists]
    draw_opts = {"ngluino_vs_status": "COLZ"}
    logy_keys = {"pt", "energy", "ht"}#, "dau_pdgid"

    n_per = max(1, args.plots_per_canvas)
    chunks = [order[i:i + n_per] for i in range(0, len(order), n_per)]
    is_pdf = args.out.lower().endswith(".pdf")
    canvases = []          # hold refs until every Print() completes

    for page, keys in enumerate(chunks):
        ncols = int(math.ceil(math.sqrt(len(keys))))
        nrows = int(math.ceil(float(len(keys)) / ncols))
        c = ROOT.TCanvas("c%d" % page, "validation page %d" % (page + 1),
                         500 * ncols, 400 * nrows)
        c.Divide(ncols, nrows)
        canvases.append(c)

        for i, key in enumerate(keys, start=1):
            c.cd(i)
            h = hists[key]
            if isinstance(h, ROOT.TH2):
                h.Draw(draw_opts.get(key, "COLZ"))
            else:
                h.SetLineWidth(2)
                h.SetLineColor(ROOT.kAzure + 2)
                h.Draw(draw_opts.get(key, "HIST"))
                ROOT.gPad.SetLogy(key in logy_keys)

            lab = ROOT.TLatex()
            lab.SetNDC(True)
            lab.SetTextSize(0.040)
            lab.SetTextFont(42)
            txt = "p_{T} > %.0f, |#eta| < %.1f" % (args.jet_pt_min, args.jet_eta_max)
            if key in ("ngluino_sx", "gluino_ndaughters", "gluino_dau_mass"):
                txt += ", %s" % args.selector
            lab.DrawLatex(0.12, 0.92, txt)
            ROOT.SetOwnership(lab, False)   # else GC'd before Print()

        if is_pdf and len(chunks) > 1:
            if page == 0:
                c.Print(args.out + "(")
            elif page == len(chunks) - 1:
                c.Print(args.out + ")")
            else:
                c.Print(args.out)
        elif len(chunks) == 1:
            c.SaveAs(args.out)
        else:
            base, ext = (args.out.rsplit(".", 1) + ["png"])[:2]
            c.SaveAs("%s_%s.%s" % (base, "_".join(keys), ext))

    print("Saved plots to %s (%d page(s), %d plots/canvas)"
          % (args.out, len(chunks), n_per))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    print("Opening file: %s" % args.infile)

    if args.branches:
        mode_branches(args)
        return

    backend = detect_backend(args.infile, args.backend)

    if args.dump:
        mode_dump(args, backend)
        return

    # v2 fix: this used `if args.dump_gluino_decay is not None`, and
    # action="store_true" yields False -- never None -- so the branch ALWAYS
    # ran and the analysis loop below was unreachable dead code.
    if args.dump_gluino_decay:
        mode_dump_decay(args, backend)
        return

    names = set(list_branches(args.infile))
    with_jets = (backend == "fwlite") or \
        (find_jet_prefix(names, args.jet_prefix) is not None)
    hists = book_histograms(with_jets)
    use_flags = not args.no_status_flags

    n_events = 0
    for ev in iter_events(args, backend):
        n_events += 1

        if with_jets and ev.has_jets:
            sel = [j for j in ev.jets
                   if j.pt() > args.jet_pt_min and abs(j.eta()) < args.jet_eta_max]
            hists["njets"].Fill(len(sel))
            ht = 0.0
            for j in sel:
                hists["pt"].Fill(j.pt())
                hists["eta"].Fill(j.eta())
                hists["phi"].Fill(j.phi())
                hists["mass"].Fill(j.mass())
                hists["energy"].Fill(j.energy())
                try:
                    hists["nconst"].Fill(j.nConstituents())
                    hists["ndaugh"].Fill(j.numberOfDaughters())
                except AttributeError:
                    pass
                ht += j.pt()
            hists["ht"].Fill(ht)

            if len(sel) >= 6:
                lead6 = sorted(sel, key=lambda j: j.pt(), reverse=True)[:6]
                best = None
                # combinations() returns both a triplet AND its complement, so
                # anchoring on jet 0 halves the loop for an identical result.
                for combo in itertools.combinations(range(6), 3):
                    if 0 not in combo:
                        continue
                    other = tuple(k for k in range(6) if k not in combo)
                    ma = invariant_mass([lead6[k] for k in combo])
                    mb = invariant_mass([lead6[k] for k in other])
                    d = abs(ma - mb)
                    if best is None or d < best[0]:
                        best = (d, ma, mb)
                if best is not None:
                    # fill BOTH triplets, not their average: the average hides
                    # the per-triplet width you compare against a reference.
                    hists["m3j"].Fill(best[1])
                    hists["m3j"].Fill(best[2])
                    hists["m3j_diff"].Fill(best[0])

        hists["ngluino"].Fill(ev.n_gluino)
        hists["ngluino_sx"].Fill(ev.n_gluino_sx)

        for gp in ev.gluinos:
            hists["gluino_mass"].Fill(gp.mass())
            hists["gluino_pt"].Fill(gp.pt())
            hists["ngluino_vs_status"].Fill(ev.n_gluino, gp.status())
            try:
                sf = gp.statusFlags()
                for bi, nm in enumerate(FLAG_NAMES_ORDERED):
                    if getattr(sf, nm)():
                        hists["flagbits"].Fill(bi)
            except Exception:
                pass

            intermediates, daughters = resolve_decay_chain(gp, use_flags)
            hists["gluino_ndaughters"].Fill(len(daughters))
            if daughters:
                hists["gluino_dau_mass"].Fill(invariant_mass(daughters))
                for d in daughters:
                    hists["dau_pdgid"].Fill(d.pdgId())

        if n_events % 200 == 0:
            print("  processed %d events..." % n_events, flush=True)

    print("Done. Processed %d events (backend=%s)." % (n_events, backend))

    order = ["njets", "pt", "eta", "phi", "mass", "energy", "nconst", "ndaugh",
             "ht", "m3j", "m3j_diff", "ngluino", "ngluino_sx",
             "ngluino_vs_status", "gluino_pt", "gluino_mass",
             "gluino_ndaughters", "gluino_dau_mass", "dau_pdgid", "flagbits"]
    draw_all(hists, args, order)

    root_out = args.root_out or (args.out.rsplit(".", 1)[0] + ".root")
    fout = ROOT.TFile(root_out, "RECREATE")
    for h in hists.values():
        h.Write()
    fout.Close()
    print("Saved histograms to %s" % root_out)


if __name__ == "__main__": main()
