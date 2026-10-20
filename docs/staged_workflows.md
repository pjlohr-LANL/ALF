# Staged ALF workflows and retained-frame budgets

The optional tools-owned runner executes a linear sequence of ordinary ALF
runs. Each stage gets independent configuration, data, models, status, journal,
logs, and Parsl resources. Stage names have no built-in scientific meaning:
ALF, GALF, temperature changes, and acquisition changes are choices in the
stage's normal configuration files.

The original `python -m alframework --master ...` workflow retains its existing
behavior when `run_control` is absent. Existing manual examples still work.

## Definition and execution

```json
{
  "final_training": true,
  "stages": [
    {"name": "alf", "master_config": "master_alf.json", "target_total_frames": 40000},
    {"name": "galf", "master_config": "master_galf.json", "target_total_frames": 60000,
     "checkpoints": [50000]}
  ]
}
```

```bash
python -m alframework.tools.staged_workflow \
  --config workflow.json --run-dir new_experiment --validate
python -m alframework.tools.staged_workflow \
  --config workflow.json --run-dir new_experiment
python -m alframework.tools.staged_workflow --run-dir new_experiment --status
python -m alframework.tools.staged_workflow \
  --config workflow.json --run-dir new_experiment --resume
```

Validation reads configuration without starting Parsl or creating the run
directory. New execution refuses an existing destination. Resume uses the
original pinned definition and verifies stage inputs/checkpoints before work
continues. Run the controller in a persistent session or suitable driver
allocation, with the intended ALF environment active and Parsl configured.

Seed generation remains a separate explicit command. If the first master
uses external H5 bootstrap replay, generate its geometry seed file before
starting. If its source H5 store already contains accepted labels, the runner
copies verified finite completed rows and trains fresh models from them.
No source model or status is imported.

An ALF-only workflow has one stage. `final_training` defaults to true; setting
it false freezes the final dataset and stops without fitting the endpoint model.
Every stage target creates an exact checkpoint. Additional checkpoints must
lie above the inherited count and at or below the current stage's target.

## What a budget means

`target_total_frames` is a cumulative number of finite, retained QM-labeled
frames, including seeds and inherited labels. A 40k stage followed by a 60k
stage selects 40k initial ALF frames and adds 20k GALF frames to the selected
lineage. It is not a target of 60k additional labels.

The budget is checked after bootstrap, after each committed H5 save, and on
startup/resume. Counts come from actual stored rows and configured scientific
properties, rather than task counts or screening estimates. Coordinates, cells
where present, and configured numeric properties must be finite. No additional
geometry or force screen is introduced by snapshot selection.

At the target, the driver stops submitting new builders and samplers. Submitted
builders finish; their unused outputs and incomplete sampler buffers are
counted without starting more dynamics. Submitted samplers finish, their
candidates receive QM labels, and the last completed QM batch is saved even if
it is smaller than `save_h5_threshold`. Outstanding training finishes too.
Parsl cleanup precedes successful stage completion and the next stage's launch.

All accepted labels remain in the stage's raw H5 store. The exact checkpoint
contains the first requested number in numeric shard order, then sorted
molecular-group order, then stored row order. Partial final groups/shards are
sliced. Overshoot stays in the raw store and is reported separately; it is not
inherited or used for final endpoint training. Independent frames sharing an
ID are retained independently.

This is an exact **selected-dataset** budget. It does not cap raw labels, QM
submission attempts, GPU hours, or wall time. Hard cost limits and drain
cancellation/timeouts are future extensions. A graceful pause waits for
submitted work and therefore also depends on working executors/resources.

## Training and scientific compatibility

Every stage starts fresh training on its initial pool and waits for complete
ensemble success before sampling. Models are not inherited. Routine ALF
retraining continues during acquisition; new routine training stops when the
stage begins draining. The next stage trains once on its predecessor's exact
checkpoint, avoiding a redundant intermediate endpoint fit.

With the default `final_training: true`, completion additionally requires a
verified ensemble trained on the exact last checkpoint. These models live in
`final_training/models`, with the input checkpoint recorded separately from
the raw stage store. Normal trainer filters and train/validation/test splits
still apply; excited-state summaries expose input and post-filter counts.

Inherited stages require compatible property names/scopes/scales, QM task entry point
and scientific settings, energy offset, and reference atom identities and
connectivity where configured. Sampler strategy, temperature, acquisition,
ML architecture, and optimizer may vary through ordinary stage configs.
Changing labeling theory requires a separate labeling workflow.

Topology IDs, atom ordering, forces, NAC arrays, stored units, arbitrary aligned
properties, and attributes are preserved. Supplied unit attributes must agree
across molecular groups and shards; absent attributes do not introduce an
inferred unit conversion. Untagged legacy rows remain untagged;
no identity mappings are fabricated. Ambiguous layouts fail explicitly.

## Paths and resources

Master paths in a workflow resolve relative to the workflow file. A master's
`master_directory: "pwd"` resolves relative to that master template's directory.
Its role configs and nested input paths resolve from that directory. Inputs
are kept read-only; ordinary ALF output paths are rebound under the new stage.
The runner copies a local Parsl module when present. Each child has an isolated
Python process and DFK, and final training uses a separate process/allocation.

For custom task outputs, add run-relative bindings to a stage:

```json
"output_paths": {
  "QM.custom_log_dir": "qm_logs",
  "sampler.custom.output_path": "sampling/diagnostics.json"
}
```

Standard data/model/status/scratch/plot paths, PySEQM logs, sampler metadata,
and force-screen audits have default bindings. Custom task code must use its
configured paths. Site-specific accounts, partitions, GPU directives, worker
environment activation, and resource quantities remain the responsibility of
the ordinary Parsl configuration. Normal task signatures and ALF return
contracts remain authoritative.

## Durability and resume

A workflow lock and per-run locks prevent competing controllers. Stage
configuration and referenced input-file fingerprints are pinned. Editing
scientific settings, checkpoints, or committed artifacts requires explicit
repair or a new workflow; resume does not silently accept the change.

The controller maintains atomic JSON status and a SQLite task journal, using
full synchronous transactions and the default rollback journal. Use a shared
filesystem supporting POSIX locks and atomic rename. Journal pickle payloads
are trusted local execution records and are not a data interchange format.

Inputs are journaled before submission and results before storage. Per-stage
namespaces keep submission identities independent of scientific frame IDs and
independent branches. H5 shards carry row provenance and transaction IDs; they
are validated, fingerprinted, flushed, and atomically published. The journal
reconciles a published shard after a crash without labeling or storing it again.
Screening accounting is committed with the storage transaction. Failed, unknown,
and successful attempt histories remain available after retries.

After an abrupt interruption, committed shards and harvested results are
reused. An unresolved submission may be recomputed because its worker might
have completed before the controller harvested the result. That uncertainty
and possible recomputation are reported. Failed backend tasks block advancement;
an explicit resume retries them after the underlying problem is repaired.
Unconverged or screened-out labels remain ordinary rejections, without automatic
seed refill.

Interrupted model directories are preserved. Recovery starts a new recorded
training attempt instead of inferring success from directory existence.
Already successful training results can be promoted from the journal without
retraining. The final fit uses the same journal and ensemble-success checks.

SIGINT/SIGTERM requests a drain and pause, and prevents the next stage from
starting. During final training the controller waits for the fit to finish,
then pauses. Resume reuses any completed final ensemble.

## Outputs and accounting

- `workflow.json`: resolved immutable plan and source fingerprints.
- `workflow_status.json`: stage outcomes, final-model receipt, and failure history.
- `stages/NN-name/control/progress.json`: accepted counts, phase membership,
  overshoot, queue state, timing, convergence/screening totals, and attempts.
- `stages/NN-name/control/journal.sqlite`: task inputs/results, attempt history,
  storage intents, and transaction receipts.
- `stages/NN-name/checkpoints/COUNT`: exact H5 selections and manifests.
- `final_training`: exact input records, training journal/status, and final model.

QM attempt counters describe controller submissions plus observed Parsl retries.
They can have unknown outcomes after a hard interruption and are distinct from
retained-frame counts. Screening reason counts can overlap. Retained overshoot
is separate from unused builder geometry and incomplete sampler buffers.

The first version supports linear complete-ALF stages and an internal final-fit
operation. The generic lifecycle boundaries can support additional stage types
later; branching workflows, model warm starts, hard QM/wall-time caps, and
cross-theory relabeling are outside this version. See the
[keto example](../examples/keto_rdkit_alf_galf/README.md) for ready configurations.
