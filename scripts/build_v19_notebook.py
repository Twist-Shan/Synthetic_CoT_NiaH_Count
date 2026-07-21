from __future__ import annotations

import json

from build_v16_1_v18_notebooks import NOTEBOOK_DIR, ROOT, build_v19


def main() -> None:
    path = NOTEBOOK_DIR / "Trace_Count_v19_Colab.ipynb"
    path.write_text(json.dumps(build_v19(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
