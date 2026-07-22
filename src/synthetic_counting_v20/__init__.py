"""Three-character-set Tiny Shakespeare counting experiments (v20)."""

import sys

# Pandas optional accelerators are not used by this project.  Declaring them
# absent avoids importing stale system-wide binary wheels in mixed notebook or
# Windows environments; the core NumPy-backed pandas path remains unchanged.
for _optional in ("pyarrow", "numexpr", "bottleneck"):
    sys.modules.setdefault(_optional, None)

from .config import V20Config, config_from_dict, preset_config  # noqa: E402

__all__ = ["V20Config", "config_from_dict", "preset_config"]
