# Config-driven KDG-ladder + FPS seeds

This prepares an unlabeled starting pool without dynamics or force-field
optimization. All scientific settings live in JSON. A reference geometry is
required because the distance bounds are centered on that structure.

From the production fork checkout, using the existing ALF environment:

```bash
python -m alframework.tools.rdkit_initialization \
  --config examples/rdkit_initialization/enol.json \
  --output-dir /path/to/new/enol_seed

python -m alframework.tools.rdkit_initialization \
  --config examples/rdkit_initialization/keto.json \
  --output-dir /path/to/new/keto_seed
```

Use a new output directory for each preparation. Existing paths are refused.
Run from this checkout so `python -m` imports the production fork; workers in
the eventual ALF run must also import this fork. The optional `topology` extra
provides RDKit; the existing environment already has it. The tool itself has
no Torch, HIPPYNN, GPU, QM, or external benchmark dependency.

## Method and editable settings

The shipped JSON files reproduce the KDG/FPS method used for the original
RDKit-seeded enol/keto runs:

- Generate **50,000 accepted geometries**, equally distributed over six levels.
- Use KDG: `useBasicKnowledge=True`, `useExpTorsionAnglePrefs=False`, random
  coordinates, box multiplier 4, seed 42, one thread. No RMSD pruning, force-field
  minimization, anchored subpopulation, or dynamics is used.
- Center 1–2 and 1–3 distance intervals on the reference, retaining at least the
  RDKit half-width. Bond/angle widening fractions are 0, 2, 4, 6, 8, and 10%.
  Ordinary hydrogen fractions are 0, 1, 2, 3, 3, and 3%; hydrogens attached to
  N/O/S use the heavy-atom fractions. Heavy-heavy 1–4 intervals widen by 0, 6,
  12, 18, 24, and 30%. These are **distance-bound fractions**, not angular noise.
  Triangle smoothing follows widening.
- Screen finite coordinates, reference bond ratios 0.85–1.50, nonbonded clearance
  at least 1.40 times summed covalent radii, no extra connection at or below 1.25
  times those radii, and minimum pair distance 0.7 Å. Screen thresholds are
  seed-specific JSON settings; they do not change global ALF sampler policy.
- Select **5,000 geometries before QM**, by independent FPS within each level.
  Features are unscaled all-atom pair distances sorted within element-pair
  types. FPS starts nearest the centroid, then uses greedy max–min distance;
  ties use the lowest pool-row index. Remainders go to lower levels first.

Required keys are `smiles` and `reference_geometry`; defaults supply the other
settings. Relative reference paths resolve against the JSON file's directory.
The examples spell out every setting so they can be edited without Python
changes. For a reduced smoke run, copy a configuration and lower `pool_size`
and `seed_size` (for example, 60 and 12). Counts need not divide evenly across
levels. `max_embedding_rounds` bounds retries; an unmet stratum quota fails
explicitly rather than silently producing a smaller pool.

## Prepared artifacts

The output directory is published only after all files have been written:

| File | Contents |
|---|---|
| `pool.h5` | Every accepted pool geometry, its stratum/source ID, and FPS rank (`-1` when unselected). |
| `seed.h5` | Selected pool rows, ordered by level and then FPS rank. |
| `reference.xyz` (or the input format's extension) | Byte-for-byte reference snapshot used by the emitted ALF topology configuration. |
| `builder_config.json` | External sequential bootstrap source, ready for the existing H5 replay builder. |
| `provenance.json` | Resolved settings, software versions, reference/code/configuration fingerprints, artifact hashes, graph mapping, embedding failures, quotas, and ALF configuration fields. |

Each H5 has one `molecule` group with `coordinates`, `species`, `_id`,
`topology_atom_ids`, `pool_index`, `strata`, and `fps_rank`. Species follow
upstream's stable atomic-number ordering. Genuine atom identities come from
one verified SMILES-to-reference graph mapping; they are preserved through
storage sorting. There are **no energies or forces** in these geometry files.
Do not copy `seed.h5` into ALF's labeled training store.

Fixed settings, thread count, and software versions reproduce arrays and
selections; H5 file bytes need not be identical across preparations. Provenance
records the actual versions and hashes. Another RDKit version may generate
different conformers even with the same random seed.

## Connect to an ordinary ALF run

Keep the run's QM, ML, electronic-state properties, ensemble size, sampler, and
resource configuration. Set the fields listed under
`provenance.json:alf_configuration.master` in its master JSON. For the defaults:

```json
{
  "builder_task": "alframework.builders.h5_replay_builder.h5_replay_builder_task",
  "builder_config_path": "/path/to/new/enol_seed/builder_config.json",
  "bootstrap_set": 5000
}
```

The local CPU entry point `h5_replay_builder_local_task` is also available when
the Parsl configuration already provides `alf_builder_executor`. Preserve the
existing `maximum_builder_structures` and batching settings. Start in a fresh
run with an empty labeled H5 store and no trained models if new labeling is
intended: normal ALF resume rules still apply.

Copy the fields under `provenance.json:alf_configuration.sampler` into the
sampler JSON, preserving other settings. They reference the prepared topology
snapshot and matching bond/distance thresholds. The tool does not rewrite an
existing master or sampler JSON. No core-driver change is required.

To match the original post-QM force screen, explicitly enable the existing
storage screen in the sampler configuration:

```json
{
  "max_force_cutoff": 16.0,
  "dataset_screening": {
    "enabled": true,
    "force": true,
    "min_distance": true,
    "topology": true
  }
}
```

The force screen checks QM atom-vector norms across every configured `F#`
state, in the run's existing units (eV/Å for these examples). It is distinct
from training's component-outlier filters and sampling's physical-force screen.
The special 1.40 nonbonded seed clearance is already enforced before selection;
it is not introduced as a new sampler option.

ALF replays each selected source row sequentially for bootstrap single-point
labeling. It stores accepted QM labels in its normal `data-0000.h5`, trains a
fresh ensemble, and subsequently draws starting structures from its own labeled
H5 shards. The external geometry pool supplies no labels or models. Geometry
preparation therefore works with any supported number of electronic states and
either existing QM backend.

## Count accounting

`pool_count` and `selected_count` in provenance describe geometry preparation,
not successful QM labels. Track convergence separately in the existing QM
diagnostics. Per-batch storage-screen totals and rejection counts are recorded in ALF's
screening output and `status.last_dataset_screening`; retained counts are the
rows actually written to the run's H5 store. An enabled storage screen also
rejects unusable/nonconverged results.

The default 5,000 selected geometries are not a promise of 5,000 retained
labels. Do not increase `bootstrap_set` to compensate for rejections: it denotes
the selected input count. There is no automatic refill. The historical 4,908
enol and 4,733 keto retained seeds depend on their original labeling and screens;
they are not guaranteed outcomes of a new preparation or a different QM method.
