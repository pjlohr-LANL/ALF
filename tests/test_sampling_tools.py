"""Dependency boundaries for shared sampling validation and screening."""
import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_sampling_tools_import_without_gpu_stack_or_sampler_modules(tmp_path):
    script = '''
import builtins
import sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'hippynn', 'nvalchemi', 'warp'} or name.startswith('alframework.samplers'):
        raise AssertionError('Unexpected optional/strategy import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from alframework.tools import sampling_validation, force_screen, dataset_screening
from alframework.tools.molecules_class import MoleculesObject
from ase import Atoms
sampling_validation.validate_sampler_debug_output(MoleculesObject(Atoms('H'), 'legacy'), {})
assert sampling_validation.configured_score({})['mode'] == 'max'
assert force_screen.validated_force_cutoff(16., positive=True) == 16.
assert not force_screen.physical_force_exceeds({}, None)
assert not any(name.split('.')[0] in {'torch','hippynn','nvalchemi','warp'} for name in sys.modules)
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
