# Excited-state test consolidation — 2026-10-19

This report records the test-only consolidation ending at `e7cbd59`; subsequent
feature tests are counted in the [RDKit audit](rdkit_initialization_audit_2026-10-19.md).

Baseline: `e1f875b` on `production_excited_states`. Changes are confined to tests,
test helpers, and documentation in the production checkout. Runtime code, public
interfaces, example settings, packaging, CI, and pytest selection remain unchanged.

## Measured reduction

| Metric | Before | After | Reduction |
|---|---:|---:|---:|
| Collected cases, including parametrization | 422 | 400 | 22 |
| Test functions | 303 | 279 | 24 |
| Test modules | 26 | 24 | 2 |
| Test-module lines | 9,544 | 8,489 | 1,055 |
| Helper/conftest lines | 175 | 401 | −226 (shared setup moved here) |
| Total test and support lines | 9,719 | 8,890 | **829 (8.5%)** |

Lines include blank lines and docstrings in every Python file below `tests`;
test modules are `test_*.py`. Shared code is included in the total so moving
setup cannot masquerade as a reduction. The 47 upstream cases remain intact;
all nine upstream test modules now match local upstream byte-for-byte.

The physical-force regression suite is unchanged apart from helper imports.
Backend-specific signs/units, root indexing, atom/force/NAC alignment, model
failure cleanup, replica-local LCM switching, and zero-candidate audits remain.
No tests were moved behind new markers, skips, exclusions, or alternate commands.

## Consolidation and coverage map

Sampling fixtures now live in `tests/helpers/fakes.py`; test modules no longer
import one another. The six fork-only storage tests and their two fixtures moved
verbatim from `test_tools.py` into `test_storage_identity.py`, restoring upstream's
test module. No storage assertions were removed.

Each removed or renamed test function is accounted for below. Names refer to
functions, not parameter instances; related scenarios retain individual parameter
IDs. Intentional removals of editable-setting snapshots are distinguished from
regressions covered elsewhere.

| Previous test | Retained coverage or reason |
|---|---|
| `test_absent_replica_candidate_limit_allows_one_trajectory_to_dominate` | `test_replica_candidate_limit_controls_trajectory_concentration`: limited/unlimited cases retain separate parameter IDs and expected parents/scores. |
| `test_absent_score_block_preserves_max_ranking` | `test_default_and_explicit_max_score_preserve_legacy_ranking`: energy- and force-dominated cases retain both default/explicit equivalence and metadata checks. |
| `test_ase_fallback_gap_diagnostics_match_raw_member_reduction` | `test_ase_gap_statistics_follow_raw_member_differences`: anticorrelated ensemble and single-calculator cases preserve gap means, deviations, model counts, and zero single-member uncertainty. |
| `test_explicit_max_mode_matches_absent_block` | `test_default_and_explicit_max_score_preserve_legacy_ranking`: energy- and force-dominated cases retain both default/explicit equivalence and metadata checks. |
| `test_replica_candidate_limit_spreads_output_across_trajectories` | `test_replica_candidate_limit_controls_trajectory_concentration`: limited/unlimited cases retain separate parameter IDs and expected parents/scores. |
| `test_single_ase_calculator_gap_deviation_is_zero` | `test_ase_gap_statistics_follow_raw_member_differences`: anticorrelated ensemble and single-calculator cases preserve gap means, deviations, model counts, and zero single-member uncertainty. |
| `test_cfg_builder_strict_batch_list_sampler_to_individual_qm` | `test_actual_driver_bootstrap_training_sampling_qm_and_retraining[cfg]`: real CFG-loader body inside a local asynchronous builder; metadata, fixed state, full batch, nested results, individual QM, and retraining. |
| `test_legacy_debug_has_no_excited_state_or_gpu_requirements` | `test_sampling_tools_import_without_gpu_stack_or_sampler_modules` already executes the identical legacy validation with GPU imports blocked. |
| `test_sampler_owned_debug_validation` | `test_sampler_debug_validation` in the tools suite; same four metadata cases. |
| `test_all_example_json_files_parse` | `test_example_paths_and_task_wiring`: five shipped master variants; JSON/path checks and actual task-signature routing, including argument identity. |
| `test_darwin_parsl_resources_are_independent_and_dynamic` | `test_example_resources_are_independent_and_start_no_allocations`: all three example directories; independent providers, no initial allocations, local executors allowed, scheduler/address probes stubbed. |
| `test_example_task_signatures_receive_every_required_input` | `test_example_paths_and_task_wiring`: five shipped master variants; JSON/path checks and actual task-signature routing, including argument identity. |
| `test_keto_reference_topology_is_valid_and_stable` | `test_example_state_sampling_and_topology_contract`: state/force declarations, QM roots, configured state selection and gap pairs, valid reference topology/species, and NAC storage without NAC training. Exact hashes, atom counts, and current disabled-gap settings are intentionally no longer frozen. |
| `test_master_configs_use_complete_six_state_contract` | `test_example_state_sampling_and_topology_contract`: state/force declarations, QM roots, configured state selection and gap pairs, valid reference topology/species, and NAC storage without NAC training. Exact hashes, atom counts, and current disabled-gap settings are intentionally no longer frozen. |
| `test_production_and_compatibility_configs_match` | Exact-setting/equality snapshots intentionally removed. Each example is independently checked by the paths/task and state/sampling contracts; thresholds, accounts, resource sizes, and independently editable configurations are not frozen. |
| `test_production_sampler_settings` | Exact-setting/equality snapshots intentionally removed. Each example is independently checked by the paths/task and state/sampling contracts; thresholds, accounts, resource sizes, and independently editable configurations are not frozen. |
| `test_six_state_cycle_forms_complete_state_specific_batches` | `test_batch_cycle_assigns_complete_state_blocks_and_repeats` and `test_batch_cycle_matches_documented_fifty_replica_boundaries`; configured example states are validated in the example contract, rather than freezing six states or fifty replicas there. |
| `test_training_schema_has_six_states_and_no_gap_targets` | `test_example_state_sampling_and_topology_contract`: state/force declarations, QM roots, configured state selection and gap pairs, valid reference topology/species, and NAC storage without NAC training. Exact hashes, atom counts, and current disabled-gap settings are intentionally no longer frozen. |
| `test_disabled_outlier_filters_retain_large_finite_values` | `test_multistate_union_is_order_independent_and_preserves_all_rows`: fixed/std/combined/disabled matrix; energy and force exclusions, per-state counts, gaps, IDs, and all row alignment. |
| `test_multistate_filter_handles_zero_variance_and_disabled_outlier_filters` | `test_multistate_zero_variance_retains_all_rows`; disabled behavior is also in the multistate matrix. |
| `test_multistate_filter_is_state_order_independent` | `test_multistate_union_is_order_independent_and_preserves_all_rows`: fixed/std/combined/disabled matrix; energy and force exclusions, per-state counts, gaps, IDs, and all row alignment. |
| `test_multistate_filter_removes_minimum_distance_before_statistics` | `test_minimum_distance_is_applied_before_multistate_statistics`. |
| `test_multistate_filter_unions_standardized_energy_and_force_outliers` | `test_multistate_union_is_order_independent_and_preserves_all_rows`: fixed/std/combined/disabled matrix; energy and force exclusions, per-state counts, gaps, IDs, and all row alignment. |
| `test_multistate_filter_unions_state_outliers_and_aligns_all_arrays` | `test_multistate_union_is_order_independent_and_preserves_all_rows`: fixed/std/combined/disabled matrix; energy and force exclusions, per-state counts, gaps, IDs, and all row alignment. |
| `test_single_state_filter_matches_production_hippynn` | `test_fixed_standardized_combined_and_disabled_filters_match_upstream` and existing component-boundary, zero-variance, and recomputed-statistics oracle cases. |
| `test_coupling_is_stored_but_never_a_training_target` | `test_example_state_sampling_and_topology_contract`: state/force declarations, QM roots, configured state selection and gap pairs, valid reference topology/species, and NAC storage without NAC training. Exact hashes, atom counts, and current disabled-gap settings are intentionally no longer frozen. |
| `test_every_example_json_parses_and_paths_resolve` | `test_example_paths_and_task_wiring`: five shipped master variants; JSON/path checks and actual task-signature routing, including argument identity. |
| `test_master_config_differs_from_summed_score_only_in_paths_and_nacr` | Exact-setting/equality snapshots intentionally removed. Each example is independently checked by the paths/task and state/sampling contracts; thresholds, accounts, resource sizes, and independently editable configurations are not frozen. |
| `test_qm_config_enables_coupling_with_the_declared_property` | `test_example_state_sampling_and_topology_contract`: state/force declarations, QM roots, configured state selection and gap pairs, valid reference topology/species, and NAC storage without NAC training. Exact hashes, atom counts, and current disabled-gap settings are intentionally no longer frozen. |
| `test_sampler_and_training_configs_match_the_summed_score_variant` | Exact-setting/equality snapshots intentionally removed. Each example is independently checked by the paths/task and state/sampling contracts; thresholds, accounts, resource sizes, and independently editable configurations are not frozen. |
| `test_sampler_inherits_the_summed_score_settings` | Exact-setting/equality snapshots intentionally removed. Each example is independently checked by the paths/task and state/sampling contracts; thresholds, accounts, resource sizes, and independently editable configurations are not frozen. |
| `test_score_gap_pairs_resolve_for_every_cycled_state` | `test_example_state_sampling_and_topology_contract`: state/force declarations, QM roots, configured state selection and gap pairs, valid reference topology/species, and NAC storage without NAC training. Exact hashes, atom counts, and current disabled-gap settings are intentionally no longer frozen. |
| `test_topology_reference_loads_from_the_example_directory` | `test_example_state_sampling_and_topology_contract`: state/force declarations, QM roots, configured state selection and gap pairs, valid reference topology/species, and NAC storage without NAC training. Exact hashes, atom counts, and current disabled-gap settings are intentionally no longer frozen. |

The batching boundary matrix retains indices 0, 49, 50, 199, and 200: initial,
last-in-block, next-block, last-in-cycle, and wrap. Removed intermediate indices
99, 100, 149, and 150 repeated the same arithmetic. Full block assignment and
cycle repetition remain separately tested. The matrix still uses a nontrivial
batch width; this is not a change to production batch defaults.

## Validation and review units

| Check | Result |
|---|---|
| Complete consolidated suite, including installed ALCHEMI and PySEQM CPU smoke tests | **400 passed**, 5 warnings, 12.68 s |
| Unchanged upstream suite with production-fork imports asserted/pinned | **47 passed**, 2 warnings, 1.21 s |
| Shared sampling fixtures and storage group | **125 passed** |
| Training/filter/tools/actual-driver group | **63 passed** |
| Final example routing/state-normalization refinements | **13 passed**, 4.87 s |
| All nine upstream test modules | Byte-identical to upstream |
| Runtime/examples/setup/CI/pytest configuration versus baseline | Unchanged |
| Cross-test-module imports and diff whitespace errors | None |

Validation used the existing `atomistic` environment, one CPU thread per
numerical library, and temporary outputs under `/ram/tmp`. The first expanded
resource check exposed a test assumption that every executor had a scheduler
provider. It was corrected to include local thread executors while checking
scheduler providers independently. Its scheduler and address-discovery probes
are now stubbed, so resource inspection is offline and the complete suite passes
inside the sandbox. Existing HIPPYNN/ASE and unavailable-NVML warnings remain.

Local review units:

- `2264657`: shared sampling fixtures and upstream storage-test restoration.
- `0f9ad85`: consolidated training-filter and actual-driver coverage.
- `71c09a1`: editable example contracts and repeated sampler scenarios.

No environments were changed, remote commits pushed, jobs submitted, or GPU
calculations launched. Upstream, prototype, and isolated integration checkouts
were left unchanged.
