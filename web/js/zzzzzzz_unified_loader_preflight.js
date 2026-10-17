import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';

function enabled(value) {
  if (Array.isArray(value)) return true; // linked input: check conservatively
  return value === true || value === 1 || ['true','on','enabled','1'].includes(String(value).toLowerCase());
}

function needsSage(output) {
  return Object.values(output || {}).some(node =>
    node.class_type === 'MuseMiniMaxH3UnifiedLoader' &&
    enabled(node.inputs?.attention_enabled ?? true) && enabled(node.inputs?.sage_attention ?? false));
}

function notice(message) {
  document.getElementById('muse-loader-preflight-notice')?.remove();
  const box = document.createElement('section');
  box.id = 'muse-loader-preflight-notice';
  box.setAttribute('role', 'alertdialog');
  box.setAttribute('aria-labelledby', 'muse-loader-preflight-title');
  box.style.cssText = 'position:fixed;top:24px;left:50%;transform:translateX(-50%);z-index:100000;width:min(560px,90vw);box-sizing:border-box;padding:20px;background:#17212d;color:#eef3f8;border:1px solid #45d7c4;border-radius:12px;box-shadow:0 8px 32px #0008;font:14px/1.5 sans-serif';
  const title = document.createElement('strong');
  title.id = 'muse-loader-preflight-title';
  title.textContent = 'SageAttention needs attention — nothing queued';
  const text = document.createElement('p');
  text.textContent = message;
  const ok = document.createElement('button');
  ok.textContent = 'OK';
  ok.style.cssText = 'padding:8px 24px;background:#ac6287;color:white;border:0;border-radius:6px;cursor:pointer';
  const previousFocus = document.activeElement;
  ok.onclick = () => { box.remove(); previousFocus?.focus?.(); };
  box.append(title, text, ok);
  document.body.append(box);
  ok.focus();
}

app.registerExtension({
  name: 'Muse.UnifiedLoader.FriendlyPreflight',
  setup() {
    if (app._museSagePreflightInstalled) return;
    app._museSagePreflightInstalled = true;
    const original = app.queuePrompt;
    app.queuePrompt = async function (...args) {
      // Inspect the executable prompt, including subgraphs and converted inputs.
      // Leave unrelated graph-serialization errors to ComfyUI's normal handler.
      let prompt;
      try { prompt = await app.graphToPrompt(); }
      catch { return original.apply(this, args); }
      if (needsSage(prompt.output)) {
        try {
          const response = await api.fetchApi('/muse_unified_loader/sage_check', {
            cache: 'no-store', signal: AbortSignal.timeout(15000),
          });
          if (!response.ok) throw new Error(`Check unavailable (HTTP ${response.status})`);
          const result = await response.json();
          if (result.ok !== true) {
            notice(result.message || 'Turn off SageAttention and High-Res Stability, or install a compatible SageAttention package, then run again.');
            return false;
          }
        } catch {
          notice('The SageAttention check could not be reached. Restart ComfyUI and refresh the browser after installing this update, then try again. No render was queued.');
          return false;
        }
      }
      return original.apply(this, args);
    };
  },
});
