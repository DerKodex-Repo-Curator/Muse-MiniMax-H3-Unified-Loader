"""Read-only UI preflight; failure is data, not a render execution error."""
import asyncio
from aiohttp import web
from server import PromptServer
from .loader_checks import check_sage
from .nodes import MuseMiniMaxH3UnifiedLoader


@PromptServer.instance.routes.get('/muse_unified_loader/sage_check')
async def sage_check(request):
    try:
        await asyncio.to_thread(check_sage, MuseMiniMaxH3UnifiedLoader._registry_class)
        result = {'ok': True}
    except Exception as exc:
        result = {'ok': False, 'message': str(exc)}
    return web.json_response(result, headers={'Cache-Control': 'no-store'})
