# Storage order and topology identity

The generic writer follows main's atomic-number sorting contract. Stable
sorting also makes equal-species ordering deterministic. Coordinates and
atomic properties use this permutation; `pair_atomic` arrays use it on their
atom axis, leaving electronic-state pair order unchanged.

Ordinary untagged molecules produce no topology metadata. When ASE atoms
explicitly contain the integer array `topology_atom_ids`, the writer requires
a permutation of `0..n_atoms-1` and stores those supplied IDs through the same
permutation. It never substitutes the sorting permutation for a supplied
mapping. A formula group cannot mix tagged and untagged rows: that would leave
some rows without a known mapping. Validation occurs before creating the H5.

Topology-aware H5 replay and ALCHEMI sampling attach canonical IDs only in
reference order. Tagged inputs are normalized before topology checks. The
explicit excited-state training loader restores coordinates, species, and
all state forces to canonical ID order before its usual fixed-composition
validation. Thus old fork shards (reference order, identity IDs) and new
sorted tagged shards can share one data pool. Source H5 files remain unchanged.
NAC arrays are preserved by storage, but are not training targets of this
energy/force interface.

For untagged legacy stores, replay retains its existing validated stable
species-sorting convention and training retains its exact atomic-number
sequence requirement. Arbitrary incompatible layouts fail; no new identity
mapping is guessed from coordinates or chemical symbols. In particular, an
untagged store cannot establish identities among equal-element atoms outside
that known legacy convention. Existing incorrectly fabricated identity tags
on genuinely reordered data cannot be repaired by this patch; the source
mapping must be supplied by its producer.

For new topology-aware bootstrap data, use the topology-aware replay builder
(or an extension that explicitly supplies canonical IDs). An ordinary CFG
builder is a generic untagged producer; it cannot infer a reference mapping.
Mixing untagged and tagged stores with incompatible atomic-number sequences
continues to fail explicitly rather than silently realigning them.
