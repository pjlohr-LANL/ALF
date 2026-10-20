"""Offline contracts for editable excited-state examples, not value snapshots."""
import importlib.util
import json
from pathlib import Path

import pytest
from ase.io import read

from alframework.ml_interfaces.excited_state_hippynn_interface import validate_excited_state_training_config
from alframework.qm_interfaces.gpu4pyscf_interface import (
    validate_gpu4pyscf_config, validate_gpu4pyscf_properties,
)
from alframework.tools.molecular_topology import load_reference_topology, validate_fixed_topology
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import validate_state_selection
from alframework.tools.sampling_validation import configured_score, configured_alchemi_gap_rows
from alframework.tools.tools import build_input_dict, load_config_file, load_module_from_string


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
VARIANTS = [
    ('excited_state_pyseqm', 'master_config.json', 'pyseqm'),
    ('keto_rdkit_alf_galf', 'master_alf.json', 'pyseqm'),
    ('keto_rdkit_alf_galf', 'master_galf.json', 'pyseqm'),
    ('excited_state_pyseqm', 'master_config_existing_h5.json', 'pyseqm'),
    ('excited_state_gpu4pyscf', 'master_config.json', 'gpu4pyscf'),
    ('excited_state_gpu4pyscf', 'master_config_summed_score.json', 'gpu4pyscf'),
    ('excited_state_gpu4pyscf_nacr', 'master_config.json', 'gpu4pyscf'),
]


@pytest.fixture(params=VARIANTS, ids=['pyseqm', 'keto-alf', 'keto-galf', 'pyseqm-replay', 'gpu4pyscf', 'gpu4pyscf-sum', 'gpu4pyscf-nacr'])
def example(request, monkeypatch):
    directory, filename, backend = request.param
    directory = EXAMPLES / directory
    monkeypatch.chdir(directory)
    master = load_config_file(str(directory / filename))
    configs = {role: load_config_file(master[key], master['master_directory'])
               for role, key in [('builder', 'builder_config_path'), ('sampler', 'sampler_config_path'),
                                 ('QM', 'QM_config_path'), ('ML', 'ML_config_path')]}
    return directory, backend, master, configs


def test_example_paths_and_task_wiring(example):
    directory, backend, master, configs = example
    assert master['QM_task'].startswith(f'alframework.qm_interfaces.{backend}_interface.')
    for path in directory.glob('*.json'):
        assert isinstance(json.loads(path.read_text()), dict), path
    molecule = MoleculesObject(read(directory / configs['sampler']['topology_check']['reference_conformer_path']), 'example')
    status = {'current_model_id': 0, 'current_training_id': 0, 'current_h5_id': 1}
    all_configs = [master, *configs.values(), status]
    feeds = {
        'builder': {'moleculeid': 'example', 'builder_config': configs['builder'], 'sampler_config': configs['sampler']},
        'sampler': {'molecule_objects': [molecule], 'sampler_config': configs['sampler'], 'ML_config': configs['ML']},
        'QM': {'molecule_object': molecule, 'QM_config': configs['QM']},
        'ML': {'ML_config': configs['ML']},
    }
    required = {'builder': {'moleculeid'}, 'sampler': {'molecule_objects', 'properties_list', 'ML_config'},
                'QM': {'molecule_object', 'properties_list', 'QM_config'},
                'ML': {'ML_config', 'h5_dir', 'properties_list', 'model_path'}}
    for role, feed in feeds.items():
        assert (directory / master[f'{role}_config_path']).is_file()
        task = load_module_from_string(master[f'{role}_task'])
        arguments = build_input_dict(task.func, [feed, *all_configs], raise_on_fail=True)
        assert required[role] <= arguments.keys(), role
        for key, value in feed.items():
            if key in arguments:
                assert arguments[key] is value, (role, key)


def test_example_state_sampling_and_topology_contract(example):
    directory, backend, master, configs = example
    properties, sampler, ml = master['properties_list'], configs['sampler'], configs['ML']
    table = validate_excited_state_training_config(ml, properties)
    states = {row['state'] for row in table}
    topology = load_reference_topology(sampler, master_directory=str(directory))
    atoms = read(topology.reference_path)
    assert validate_fixed_topology(atoms, topology).valid
    assert ml['n_atoms'] == len(atoms)
    assert set(atoms.numbers) <= set(ml['network_params']['possible_species'])
    trainable = {row['energy_key'] for row in table} | {row['force_key'] for row in table}
    assert trainable == {f'{prefix}{state}' for state in states for prefix in ('sE', 'F')}
    assert 'nacr' not in trainable
    if backend == 'gpu4pyscf':
        options = validate_gpu4pyscf_config(configs['QM'])
        qm_table, _ = validate_gpu4pyscf_properties(properties, nroots=options['nroots'], compute_nacr=options['compute_nacr'])
        assert {row['state'] for row in qm_table} == states
        assert ('nacr' in properties) == options['compute_nacr']
    selection = validate_state_selection(sampler, properties)
    selected_states = selection[1] if selection else [0]
    assert set(selected_states) <= states
    for state in selected_states:
        score = configured_score(sampler, properties, state)
        diagnostics, seeking = configured_alchemi_gap_rows(sampler, properties, state)
        rows = score['gap_rows'] + diagnostics + seeking['rows']
        assert all({row['lower_state'], row['upper_state']} <= states for row in rows)
        if sampler.get('score', {}).get('gap_pairs') == 'selected_adjacent':
            pairs = {(row['lower_state'], row['upper_state']) for row in score['gap_rows']}
            assert pairs == {pair for pair in ((state-1, state), (state, state+1)) if set(pair) <= states}


@pytest.mark.parametrize('directory', sorted({variant[0] for variant in VARIANTS}))
def test_example_resources_are_independent_and_start_no_allocations(directory, monkeypatch):
    from parsl.executors.high_throughput import executor as executor_module
    from parsl.providers import SlurmProvider

    def scheduler_probe(self, command, **kwargs):
        assert command == 'sacct -X'
        return 0, '', ''

    monkeypatch.setattr(executor_module, 'get_all_addresses', lambda: {'127.0.0.1'})
    monkeypatch.setattr(SlurmProvider, 'execute_wait', scheduler_probe)
    spec = importlib.util.spec_from_file_location(f'example_resources_{directory}', EXAMPLES / directory / 'parsl_configs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config_darwin if hasattr(module, 'config_darwin') else module.config
    executors = {executor.label: executor for executor in config.executors}
    assert {'alf_ML_executor', 'alf_sampler_executor', 'alf_QM_executor'} <= executors.keys()
    providers = [executor.provider for executor in executors.values() if hasattr(executor, 'provider')]
    assert providers
    assert len({id(provider) for provider in providers}) == len(providers)
    assert all(provider.init_blocks == provider.min_blocks == 0 for provider in providers)


@pytest.mark.parametrize('filename,modes',[
    ('workflow_alf_only.json',[False]),('workflow_alf_galf.json',[False,True])])
def test_keto_staged_workflow_strategy_contract(filename,modes):
    from alframework.tools.staged_workflow import definition,source_master
    plan=definition(EXAMPLES/'keto_rdkit_alf_galf'/filename)
    actual=[]
    for stage in plan['stages']:
        master,configs,_=source_master(stage['master_config'])
        actual.append(configs['sampler']['gap_seeking']['enabled'])
        assert not configs['ML']['gap_targets']['enabled']
        assert configs['sampler']['score']['w_gap_uncertainty']==0.
    assert actual==modes
    assert plan['final_training']
