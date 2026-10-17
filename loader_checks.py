"""Small, model-free checks before loading the H3 stack."""
import importlib


def apply_pinned_limit(mm, choice):
    current = mm.MAX_PINNED_MEMORY
    # Module-scoped ownership survives multiple loader instances. Respect changes
    # made by another component instead of restoring a stale value over them.
    state = getattr(mm, '_muse_pinned_limit_state', None)
    if state is None or current != state['applied']:
        state = {'original': current, 'applied': current}
    original = state['original']
    if str(choice).startswith('Default'):
        target = original
    elif choice in ('16 GB', '14 GB', '12 GB', '10 GB', '8 GB', '6 GB'):
        target = min(original, int(choice.split()[0]) * 1024**3) if original > 0 else original
    else:
        raise ValueError(f'Unknown pinned-memory setting: {choice}')
    mm.MAX_PINNED_MEMORY = target
    state['applied'] = target
    mm._muse_pinned_limit_state = state
    return target


def check_sage(registry):
    remedy = ('Unified Loader: SageAttention cannot be enabled. Turn OFF MiniMax H3 '
              'Memory-Efficient Sage Attention (and High-Res Stability) to continue '
              'without this patch; keep Low-VRAM Attention and Chunk Feed-Forward ON. '
              'Running without Sage may require more VRAM. Alternatively install a '
              'SageAttention build matching your PyTorch, CUDA and GPU, then restart. ')
    try:
        core = importlib.import_module('sageattention.core')
        for name in ('per_thread_int8_triton', 'per_warp_int8_cuda',
                     'per_block_int8_triton', 'per_channel_fp8',
                     'get_cuda_arch_versions', 'attn_false'):
            if not callable(getattr(core, name, None)):
                raise RuntimeError(f'SageAttention is missing {name}')
        patch = registry('MiniMaxH3MemoryEfficientSageAttentionPatch')
        execute = getattr(patch, 'execute', None)
        if not callable(execute):
            raise RuntimeError('KJNodes Sage patch does not expose the expected execute API')
        scope = getattr(getattr(execute, '__func__', execute), '__globals__', {})
        if '_cuda_archs' in scope:
            archs = scope['_cuda_archs']
            if not archs:
                raise RuntimeError('KJNodes could not initialise SageAttention GPU support')
            arch = archs[0]
            kernels = {'sm75': None, 'sm80': '_qattn_sm80', 'sm86': '_qattn_sm80',
                       'sm89': '_qattn_sm89', 'sm90': '_qattn_sm90',
                       'sm120': '_qattn_sm89', 'sm121': '_qattn_sm89'}
            if arch not in kernels:
                raise RuntimeError(f'Unverified GPU architecture for this H3 Sage patch: {arch}')
            kernel = kernels[arch]
            if kernel and kernel in scope and scope[kernel] is None:
                raise RuntimeError(f'KJNodes could not load the SageAttention kernel for {arch}')
        if '_ck' in scope and scope['_ck'] is None:
            raise RuntimeError('KJNodes could not import the required ComfyUI H3 support')
    except Exception as exc:
        raise RuntimeError(remedy + f'Detail: {type(exc).__name__}: {exc}') from exc
