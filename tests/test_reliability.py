import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('checks', root/'loader_checks.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
GB = 1024**3
mm = SimpleNamespace(MAX_PINNED_MEMORY=20*GB)
for choice, expected in [('8 GB',8), ('16 GB',16), ('Default (ComfyUI)',20),
                         ('6 GB',6), ('Default (40% of RAM)',20)]:
    assert c.apply_pinned_limit(mm,choice) == expected*GB
for disabled in [0,-1]:
    mm = SimpleNamespace(MAX_PINNED_MEMORY=disabled)
    assert c.apply_pinned_limit(mm,'16 GB') == disabled
    assert c.apply_pinned_limit(mm,'Default (ComfyUI)') == disabled
mm = SimpleNamespace(MAX_PINNED_MEMORY=4*GB)
assert c.apply_pinned_limit(mm,'16 GB') == 4*GB
c.apply_pinned_limit(mm,'6 GB')
mm.MAX_PINNED_MEMORY = 0  # another component disables pinning
assert c.apply_pinned_limit(mm,'Default (ComfyUI)') == 0
try:
    c.apply_pinned_limit(mm,'garbage')
except ValueError:
    pass
else:
    raise AssertionError('Invalid limit accepted')

symbols = ('per_thread_int8_triton','per_warp_int8_cuda','per_block_int8_triton',
           'per_channel_fp8','get_cuda_arch_versions','attn_false')
core = SimpleNamespace(**{n:lambda:None for n in symbols})
def make_patch(archs, kernel=object()):
    scope = {'_cuda_archs':archs,'_qattn_sm89':kernel,'_ck':object()}
    exec('def execute(model): pass',scope)
    return SimpleNamespace(execute=scope['execute'])
with patch.object(c.importlib,'import_module',return_value=core):
    c.check_sage(lambda name:make_patch(['sm120']))
    for p in [make_patch(None),make_patch(['sm120'],None),make_patch(['unknown'])]:
        try:
            c.check_sage(lambda name:p)
        except RuntimeError as e:
            assert 'Turn OFF' in str(e) and 'Detail:' in str(e)
        else:
            raise AssertionError('Broken Sage accepted')
with patch.object(c.importlib,'import_module',side_effect=ImportError('missing library')):
    try:
        c.check_sage(lambda name:None)
    except RuntimeError as e:
        assert 'missing library' in str(e)
    else:
        raise AssertionError('Missing library accepted')

# Execute the actual master load method with a fake parent: verify ordering and bypass.
tree = ast.parse((root/'nodes_attention_master.py').read_text())
cls = next(n for n in tree.body if isinstance(n,ast.ClassDef))
load = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='load')
events=[]
class Base:
    def load(self,**values):
        events.append('models')
        return {'result':('ok',)}
scope={'Base':Base,'check_sage':lambda registry:events.append('check'),
       '_attention_boolean':lambda v,n:v}
testclass=ast.ClassDef(name='Loader',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[load],decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[testclass],type_ignores=[])),'<master>','exec'),scope)
obj=scope['Loader']();obj._registry_class=lambda name:None
obj.load(sage_attention=True);assert events==['check','models']
events.clear();obj.load(sage_attention=False);assert events==['models']
events.clear();obj.load(sage_attention=True,attention_enabled=False);assert events==['models']
def fail(registry): raise RuntimeError('early stop')
scope['check_sage']=fail;events.clear()
try: obj.load(sage_attention=True)
except RuntimeError: pass
assert events==[]
for f in root.glob('*.py'): ast.parse(f.read_text())
print('PASS: limits, restoration, external changes, disabled states, Sage failures, preflight order, bypass, syntax')
