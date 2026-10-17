"""Master attention switch for the Muse MiniMax H3 unified loader."""

from .nodes_userfriendly import MuseMiniMaxH3UnifiedLoader as _FriendlyLoader
from .high_res_stability import apply_high_res_stability
from .loader_checks import check_sage


def _attention_boolean(value, name):
    """Accept old text toggles without treating the string 'Off' as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "on", "enabled", "1"):
            return True
        if text in ("false", "off", "disabled", "0"):
            return False
    elif isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be an ON/OFF boolean, got {value!r}")


class MuseMiniMaxH3UnifiedLoader(_FriendlyLoader):
    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        for name, tooltip in (
            ("sage_attention", "Optional H3 memory-efficient Sage patch. Requires compatible SageAttention, PyTorch/CUDA and GPU support. Checked before loading models. OFF does not disable ComfyUI launch-time SageAttention."),
            ("attention_backend", "Per-model backend override. Off means no override from this control; it does not disable launch-time Sage or the MiniMax-specific Sage toggle."),
        ):
            spec = schema["required"].get(name)
            if spec:
                options = dict(spec[1]) if len(spec) > 1 else {}
                options["tooltip"] = tooltip
                if name == "sage_attention":
                    options.update(default=False, label_on="ON", label_off="OFF")
                schema["required"][name] = (spec[0], options)
        # Append for saved-workflow compatibility. The fascia repositions this
        # control visually at the top of the Attention & Memory card.
        schema["required"]["attention_enabled"] = (
            "BOOLEAN",
            {
                "default": True,
                "label_on": "Attention optimisations ON",
                "label_off": "Attention optimisations OFF",
                "tooltip": "Master bypass for every attention and feed-forward patch in this loader.",
            },
        )
        # Append only: old widget indices remain unchanged.
        schema["required"]["models_to_load"] = (
            ["Both / Mixed", "Reference only", "First/Last only"],
            {"default": "Both / Mixed", "tooltip": "Skip loading the unused transformer and its LoRA/attention patches. Match Director mode; Both is required for mixed timelines."},
        )
        # Keep the input position and legacy enum for saved workflows.
        # The runtime helper tracks the original process-wide limit.
        schema["required"]["pinned_memory_cap"] = (
            ["Default (ComfyUI)", "16 GB", "14 GB", "12 GB", "10 GB", "8 GB", "6 GB", "Default (40% of RAM)"],
            {"default": "Default (ComfyUI)", "tooltip": "Process-wide page-locked RAM limit. Default restores the limit captured before Muse changed it; the limit depends on your OS and ComfyUI settings. Numeric caps never exceed that limit or enable disabled pinning. Applies when this loader runs; multiple loaders share the limit. The old 40% label is retained for saved workflows."},
        )
        schema["required"]["high_res_stability"] = (
            "BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF",
            "tooltip": "Packs H3 Sage attention inputs into contiguous memory before kernel execution, addressing strided-input issues. Requires SageAttention and attention optimisations ON. May allocate extra VRAM; leave OFF unless needed."},
        )
        # Optional and appended: existing saved widgets retain their positions.
        schema.setdefault("optional", {})["extra_loras"] = (
            "STRING", {"default": "[]", "multiline": False},
        )
        return schema

    def _patch_attention(self, model, values):
        if not values.get("attention_enabled", True):
            return model
        model = super()._patch_attention(model, values)
        if values.get("high_res_stability", False) and values.get("sage_attention", False):
            model = apply_high_res_stability(model)
        return model

    def load(self, **values):
        for name in ("attention_enabled", "sage_attention", "high_res_stability",
                     "low_vram_attention", "chunk_feed_forward"):
            if name in values:
                values[name] = _attention_boolean(values[name], name)
        if values.get("attention_enabled", True) and values.get("sage_attention", False):
            check_sage(self._registry_class)
        result = super().load(**values)
        if values.get("high_res_stability", False):
            active = values.get("attention_enabled", True) and values.get("sage_attention", False)
            status = result["result"][-1] + (" High-Res Stability ON (contiguous Sage inputs)." if active else " High-Res Stability inactive: enable attention optimisations and Sage Attention.")
            result["result"] = (*result["result"][:-1], status)
            result["ui"] = {"text": [status]}
        return result


NODE_CLASS_MAPPINGS = {"MuseMiniMaxH3UnifiedLoader": MuseMiniMaxH3UnifiedLoader}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseMiniMaxH3UnifiedLoader": "Muse MiniMax H3 Unified Loader (POC)"
}
