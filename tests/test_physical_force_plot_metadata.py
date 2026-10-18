import pytest

from alframework.tools.excited_state_iteration_plot_inputs import _candidate_record


@pytest.mark.parametrize('modern', [False, True])
def test_plot_records_distinguish_legacy_propagation_and_physical_screening(modern):
    candidate = {'fmax':5.}
    if modern:
        candidate.update(sampler_metadata_schema_version=2, force_screen_source='physical_state_mean',
                         force_screen_scope='all_states', propagation_fmax=35., gap_bias_fmax=30.,
                         physical_fmax_by_state={'0':5.,'1':4.})
    record = _candidate_record(parent_molecule_id='parent', generation=0, rank=0,
        candidate=candidate, uncertainty_scope='selected_state', uncertainty_aggregate='rms')
    assert record['fmax'] == 5.
    assert record['force_screen_source'] == ('physical_state_mean' if modern else 'propagation_legacy')
    assert record['propagation_fmax'] == (35. if modern else 5.)
    assert record['sampler_metadata_schema_version'] == (2 if modern else 1)
