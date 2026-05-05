# ALF Parsl Resource Model

This note records how ALF uses Parsl to place builder, sampler, QM, and ML work on compute resources. It is written for developers expanding ALF jobs to more nodes, especially for the excited-state PySEQM workflow on Darwin.

## ALF Master vs Parsl Workers

The ALF master process is the Python process launched with:

```bash
python -m alframework --master <master_config.json>
```

The master process runs the orchestration in `alframework/__main__.py`. It owns:

- config loading and dynamic task resolution
- ALF queue bookkeeping through `parsl_task_queue`
- `status.txt` updates
- HDF5 shard writing
- ML model rollover after training succeeds
- submission of Parsl app futures for builder, sampler, QM, and ML stages

The master does not perform most expensive stage work itself. Stage functions are Parsl apps. When ALF calls a stage task, Parsl sends that task to a worker process attached to the executor named by the task decorator.

Parsl workers are persistent worker processes. A QM frame is not a separate Slurm job in the in-allocation setup. Instead, ALF submits many QM futures to the existing `alf_QM_executor` worker pool.

## Stage-To-Executor Mapping

ALF stage placement is primarily controlled by the Parsl app decorator on each task function:

```python
@python_app(executors=["alf_QM_executor"])
def qm_task(...):
    ...
```

Common ALF executor labels are:

- `alf_sampler_executor`: builder and sampler tasks
- `alf_QM_executor`: QM labeling tasks
- `alf_ML_executor`: ML training tasks

Builders usually run on the sampler executor because there is no separate builder executor in the current core setup. In bootstrap, builder outputs go directly to QM. After bootstrap, builder outputs go to sampler, sampler outputs go to QM, QM outputs are saved to HDF5, and ML retraining is launched from accumulated HDF5 shards.

`alframework/__main__.py` can also auto-append standby executors. If a task declares `alf_QM_executor` and the active Parsl config also defines `alf_QM_standby_executor`, ALF appends the standby executor label to that task. This behavior is executor-label based; it does not change the task API.

## Current Darwin 3-Node In-Allocation Model

The current small excited-state Darwin test uses:

- `examples/excited_state_pyseqm_small_test/submit_3node_darwin_ml4chem_test.slurm`
- `alframework/parsl_resource_configs/darwin_3node_ml4chem_inalloc.py`

This is an in-allocation model:

1. Slurm reserves the full allocation first with `#SBATCH --nodes=3`.
2. The ALF master starts inside that allocation.
3. Parsl uses `LocalProvider`, not `SlurmProvider`.
4. The custom launcher uses `srun` to start worker pools on specific nodes already assigned to the job.
5. Parsl does not submit new Slurm jobs and cannot grow beyond the original allocation.

For this model, the practical rule is: request all nodes and hardware up front. If the job needs 3 GPU nodes, the outer Slurm script must request 3 GPU nodes. Parsl only subdivides and uses the nodes it already has.

## Current Node Placement

`darwin_3node_ml4chem_inalloc.py` reads `SLURM_NODELIST` and assigns the first three allocated nodes:

```text
allocated node 0 -> ALF master process plus alf_sampler_executor
allocated node 1 -> alf_QM_executor
allocated node 2 -> alf_ML_executor
```

The current executor layout is:

```text
alf_sampler_executor:
  provider: LocalProvider
  launcher: srun pinned to node 0
  max_workers_per_node: 4
  available_accelerators: 4

alf_QM_executor:
  provider: LocalProvider
  launcher: srun pinned to node 1
  max_workers_per_node: 4
  available_accelerators: 4

alf_ML_executor:
  provider: LocalProvider
  launcher: srun pinned to node 2
  max_workers_per_node: 1
  available_accelerators: 4
```

`max_workers_per_node` controls how many Parsl worker processes are started for that executor block. `available_accelerators=4` tells Parsl there are four GPU accelerator slots available to assign. Parsl masks/pins worker processes to accelerators, so code running inside a worker may only see one CUDA device even though the physical node has four GPUs.

That GPU masking matters. Worker code should choose from the CUDA devices visible inside the worker, not blindly index the physical node GPU count. This is why PySEQM device selection should use `torch.cuda.device_count()` inside the worker process.

The ML executor intentionally starts one Parsl worker in the current 3-node test. The excited-state ML task then trains the ensemble internally through multiprocessing, using `gpus_per_node` to split ensemble members across GPUs on that ML node. This is different from launching one Parsl ML task per ensemble member.

## Queue-Expanding SlurmProvider Model

ALF also contains older resource configs such as:

- `alframework/parsl_resource_configs/darwin.py`
- `alframework/parsl_resource_configs/chicoma.py`

Those configs use `SlurmProvider`. In that model, Parsl can submit Slurm blocks itself:

```python
provider=SlurmProvider(
    partition,
    init_blocks=0,
    min_blocks=0,
    max_blocks=2,
    nodes_per_block=1,
    ...
)
```

Here, `max_blocks` is the upper bound on how many scheduler blocks Parsl may request for that executor. `nodes_per_block` is the size of each block. If workload grows and Parsl scaling decides more workers are needed, Parsl may submit additional Slurm jobs up to `max_blocks`.

This is the model where Parsl can expand through the queue. It only happens if the active Parsl config uses a scheduler provider such as `SlurmProvider` with nonzero `max_blocks`. It does not happen with `LocalProvider`.

## Which Model To Use

Use the in-allocation model when:

- debugging task correctness
- validating new builder, QM, sampler, or ML interfaces
- needing predictable node placement
- needing to avoid queue latency while many short QM tasks run
- testing a fixed small number of GPU nodes

Use the `SlurmProvider` queue-expanding model when:

- task correctness is already stable
- dynamic scaling is more valuable than fixed placement
- queue wait time and scheduler policy are acceptable
- the cluster allows a workflow process to submit additional jobs
- failures from scheduler scaling are easier to tolerate and debug

For the current Darwin excited-state debugging work, the in-allocation model is the right default. It makes the resource boundary explicit: the outer Slurm job owns all nodes, and Parsl starts workers inside those nodes.

## Scaling Notes For More Nodes

There are two clean ways to scale beyond the current 3-node test.

### Expand The In-Allocation Layout

Request more nodes in the outer Slurm script, then update the in-allocation Parsl config to assign additional nodes to executor blocks.

For example, a future 5-node fixed layout could be:

```text
node 0 -> ALF master + sampler/builder workers
node 1 -> QM workers
node 2 -> QM workers
node 3 -> sampler workers
node 4 -> ML workers
```

That requires the Parsl config to create more executor blocks or additional executor labels with `LocalProvider` launchers pinned to the chosen node names. The current 3-node config has `max_blocks=1` for each executor and manually pins one block per executor.

This model is simple to reason about, but it is not elastic. If the outer Slurm job requests 5 nodes, the workflow has at most those 5 nodes.

### Move To SlurmProvider Scaling

Create or adapt a Parsl config that uses `SlurmProvider` for the desired executors. Then set:

- scheduler partition/QOS/account fields
- `nodes_per_block`
- `max_blocks`
- `walltime`
- `worker_init`
- launcher behavior

In this model, the ALF master can run in a smaller initial job or suitable service context, while Parsl submits worker blocks as scheduler jobs. This can scale farther, but it introduces queue latency and more failure modes.

## Gotchas

- The master node can also run workers. In the current Darwin test, node 0 runs the ALF master and the sampler/builder worker pool.
- `LocalProvider` cannot exceed the original Slurm allocation. It only starts local/in-allocation workers.
- `SlurmProvider` can submit queue jobs only when the active Parsl config uses it and `max_blocks` allows it.
- Parsl workers are persistent. ALF does not launch one Slurm job per molecule or per QM frame in the in-allocation setup.
- Builder tasks and sampler tasks both usually use `alf_sampler_executor`.
- ML ensemble training may use multiprocessing inside one Parsl ML task, so `max_workers_per_node` for the ML executor is not always the number of ensemble models trained at once.
- Parsl accelerator pinning can mask GPUs. Use visible-device counts inside worker code.
- The custom Darwin in-allocation launcher writes `cmd_$JOBNAME.sh` instead of `cmd_$SLURM_JOB_NAME.sh` to avoid executor launch-script collisions.
- Cleanup scripts for test reruns should not delete seed datasets such as `seed_dataset_small/`; deleting seed files makes replay builders fail before QM launch.
- `status.txt` restart state only tracks coarse progress. It does not reconstruct in-flight Parsl queues after a failed run.

## Quick Diagnostic Checklist

If no QM tasks launch:

1. Check builder status first. QM cannot start until builders return structures.
2. Inspect `runinfo/*/parsl.log` for failed builder task tracebacks.
3. Confirm seed replay files exist if using `source_priority: seed_all_once`.
4. Confirm the active Parsl config has worker logs for the executor expected to run the task.

If workers never register:

1. Check `runinfo/*/submit_scripts/*.out` and `*.err`.
2. Confirm the command file names are unique per executor block.
3. Confirm `srun` nodelists refer to nodes actually present in the outer Slurm allocation.
4. Confirm the worker environment can import `alframework`.

If GPUs behave inconsistently:

1. Check worker logs for accelerator pinning.
2. Remember each worker may see only one CUDA device.
3. Prefer worker-visible GPU selection over physical-node GPU indexing.
