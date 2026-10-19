"""Dependency boundaries for shared sampling validation and screening."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest
from ase import Atoms
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampling_validation import validate_sampler_debug_output


ROOT = Path(__file__).resolve().parents[1]


def test_sampling_tools_import_without_gpu_stack_or_sampler_modules(tmp_path):
    script = '''
import builtins
import sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] == 'rdkit':
        raise ModuleNotFoundError('RDKit deliberately unavailable')
    if name.split('.')[0] in {'torch', 'hippynn', 'nvalchemi', 'warp'} or name.startswith('alframework.samplers'):
        raise AssertionError('Unexpected optional/strategy import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from alframework.tools import sampling_validation, force_screen, dataset_screening, rdkit_initialization
try:
    rdkit_initialization._rdkit()
except ImportError as exc:
    assert 'topology extra' in str(exc)
else:
    raise AssertionError('Missing RDKit was not reported')
from alframework.tools.molecules_class import MoleculesObject
from ase import Atoms
sampling_validation.validate_sampler_debug_output(MoleculesObject(Atoms('H'), 'legacy'), {})
assert sampling_validation.configured_score({})['mode'] == 'max'
assert force_screen.validated_force_cutoff(16., positive=True) == 16.
assert not force_screen.physical_force_exceeds({}, None)
assert not any(name.split('.')[0] in {'torch','hippynn','nvalchemi','warp','rdkit'} for name in sys.modules)
'''
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_tools_have_no_imports_from_sampler_implementations():
    for path in (ROOT/'alframework/tools').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or '']
                if node.module == 'alframework':
                    modules.extend('alframework.' + alias.name for alias in node.names)
            assert not any(name == 'alframework.samplers' or name.startswith('alframework.samplers.')
                           for name in modules), str(path)
    assert not (ROOT/'alframework/samplers/force_screen.py').exists()
    assert not (ROOT/'alframework/samplers/debug_validation.py').exists()


@pytest.mark.parametrize('metadata,passes', [
    ({'sampler_device':'cuda:0','selected_state':1,'topology_valid':True}, True),
    ({'sampler_device':'cpu','selected_state':1,'topology_valid':True}, False),
    ({'sampler_device':'cuda:0','selected_state':0,'topology_valid':True}, False),
    ({'sampler_device':'cuda:0','selected_state':1,'topology_valid':False}, False),
])
def test_sampler_debug_validation(metadata, passes):
    mol = MoleculesObject(Atoms('H'), 'debug')
    mol.update_metadata(metadata)
    cfg = {'alchemi_baoab': {'strict_gpu':True}, 'model_mode':'excited_state',
           'state_selection':{'mode':'fixed','state':1}, 'topology_check':{'enabled':True}}
    if passes:
        validate_sampler_debug_output(mol, cfg)
    else:
        with pytest.raises(AssertionError):
            validate_sampler_debug_output(mol, cfg)
