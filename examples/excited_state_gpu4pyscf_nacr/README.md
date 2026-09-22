# Excited-state GPU4PySCF active learning with NACR labeling

This example labels each molecule with state energies, forces, **and**
excited-excited nonadiabatic coupling vectors (NACR), while the machine-learned
potential still trains on energies and forces only.

The point is to avoid relabeling later. Recomputing NACR over a finished dataset
means re-running every CAM-B3LYP TDA calculation; computing it during active
learning costs roughly 55% more per molecule and leaves the coupling in the
HDF5 shards for later dynamics work.

Everything except the NACR additions is inherited from
`examples/excited_state_gpu4pyscf`'s summed-score variant: same keto
acetylacetone molecule, same six states, same CAM-B3LYP/6-31G* TDA method, same
`-9405.0 eV` energy offset.

## What is computed versus what is learned

| | Computed and stored | Learned |
| --- | --- | --- |
| `sE0` … `sE5` state energies | yes | **yes** |
| `F0` … `F5` state forces | yes | **yes** |
| `nacr` coupling vectors | yes | **no** |

The split is structural, not a setting you could accidentally flip. ALF's
excited-state trainer derives its output heads by *pattern-matching* property
names: `derive_state_property_table` and `derive_gap_property_table` in
`alframework/tools/excited_state_tools.py` recognize only `sE#`, `F#`, and
`dE##`. `build_excited_state_training_graph` then creates one energy head and
one gradient node per matched state, and the loss is composed from those nodes
alone. `load_excited_state_h5_arrays` likewise reads only those names and errors
only when one is *missing* — extra datasets in a shard are never read.

Because `nacr` matches none of those patterns, no head, no loss term, and no
data load exists for it. `tests/test_excited_state_hippynn_interface.py`
asserts this directly: a shard containing `nacr` loads byte-identically to one
without it.

## Configuration

Two settings differ from the summed-score variant:

```jsonc
// QM_config.json
"compute_nacr": true

// master_config.json properties_list
"nacr": ["nacr", "pair_atomic", 1.0]
```

Both are required together. `validate_gpu4pyscf_properties` rejects the flag
without the property (coupling with nowhere to store it) and the property
without the flag (storage for data never produced).

`pair_atomic` is a storage scope added for this: the array is
`[pairs, atoms, 3]`, so the atom reordering applies to axis 1. The existing
`atomic` scope indexes axis 0 and would silently permute the pair axis instead.

Inherited from the summed-score variant, unchanged: `uncertainty_policy:
"continue"`, `score.mode: "sum"` with weights 1.0 / 1.0 / 5.0 and
`gap_pairs: "selected_adjacent"`, `return_top_k: 100`,
`max_candidates_per_replica: 10`, `Ncheck: 10`, `save_h5_threshold: 500`, and
the production network (`n_features` 145, `n_sensitivities` 20).

## Shard layout

For `nroots: 5` and 15 atoms, each molecular group gains:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `nacr` | `[structures, 10, 15, 3]` | coupling vectors, Angstrom^-1, ETF-scaled |
| `nac_pairs` | `[10, 2]` | the state pair for each slice of axis 1 |

`nac_pairs` is written once per group rather than per frame, like `species`, so
a shard describes its own pair ordering and analysis code never recomputes it.
Mixing pair orderings within one group raises rather than silently producing
ambiguous slices.

Pairs are the ordered excited-excited combinations:

```
(1,2) (1,3) (1,4) (1,5) (2,3) (2,4) (2,5) (3,4) (3,5) (4,5)
```

Count is `C(nroots, 2)` — 10 for `nroots: 5`, 15 for 6 — so both storage and
solver cost grow quadratically in state count.

## Two properties of the stored coupling

**The sign is arbitrary.** Each pair's NACR carries an arbitrary global sign
from the solver. ALF stores the raw sign with no phase alignment, recorded as
`nacr_sign_convention: "raw_solver_arbitrary_global_sign"` in metadata. Nothing
trains on NACR, so this is correct — but downstream analysis comparing
geometries or runs must resolve the per-pair sign itself.

**The S1 force changes provenance.** With `compute_nacr` enabled, the NAC solver
runs with `grad_state=1` and returns the S1 gradient as a byproduct, so the
direct TDA gradient loop starts at S2. Metadata records which path produced it
via `s1_gradient_source` (`"nac_solver"` or `"tda_gradient"`). This is why the
option is opt-in rather than always on: it changes how an existing quantity is
obtained, not only what is added.

## Cost

Measured on 30-atom indigo at 6-31G* on an A100
(`Projects/ml_4_excited_states/TDDFT/benchmark_results/17238516`, three
fresh-process runs per basis, gpu4pyscf 1.7.4):

| | Without NACR | With NACR |
| --- | --- | --- |
| Total per molecule | 32.5 s | 50.3 s (**1.55x**) |
| Peak GPU memory | — | 22.7 GiB of 40 GiB |

Breakdown with coupling: SCF 25.4 s, TDA 21.1 s, NAC + S1 kernel 28.1 s,
S2–S5 gradients 17.8 s. Keto acetylacetone is half indigo's atom count so
absolute times are lower, but expect a similar ratio.

Two capacity notes: with four QM workers per node, 22.7 GiB peak per worker on
a 30-atom molecule is worth watching; and `return_top_k: 100` queues up to 100
of these calculations per sampler task.

Storage grows about 1.5x. Per frame at `nroots: 5` and 15 atoms, `nacr` is
10 x 15 x 3 = 450 float64 values against 6 x 15 x 3 = 270 for all forces.

## Running

Needs a labeled seed and a trained ensemble, same as the parent example:

```text
h5store/data-0000.h5
models/model-0000/
```

With both present and no `status.txt`, the driver skips bootstrap and initial
training and samples immediately. Then:

```bash
sbatch submit_darwin.slurm
```

Check the setup without a full run:

```bash
cd /vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/examples/excited_state_gpu4pyscf_nacr
python -m alframework --master master_config.json --test_qm
```

That writes `qm_test.h5`. Confirm `nacr` has shape
`[frames, 10, atoms, 3]`, `nac_pairs` is `[10, 2]` matching the list above, all
values are finite, and the `sE#`/`F#` datasets are unchanged in shape. Metadata
should record `compute_nacr`, `nac_scope`, `nacr_units`, `nac_pairs`, and
`s1_gradient_source`.

Before committing significant compute, cross-check the physics against the
benchmark: run one geometry through both paths with identical settings and
compare per-pair NACR L2 norms against
`nac.pair_l2_norms_angstrom_inverse` in
`benchmark_results/17238516/6-31g-star/run_1.json`, allowing for the arbitrary
per-pair sign.

Use a separate run directory from the other examples. Shards written here carry
coupling datasets that the others do not, and mixing them would produce a
dataset where only some frames have NACR.

## Turning coupling off

Set `compute_nacr: false` in `QM_config.json` **and** remove `nacr` from
`properties_list`. Both together, since the two are cross-validated. That
yields the summed-score variant's behavior exactly.
