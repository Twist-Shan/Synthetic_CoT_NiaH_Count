from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
DRIVE_TAG = "google-drive-login"

DRIVE_MARKDOWN = {
    "cell_type": "markdown",
    "id": "google-drive-login-heading",
    "metadata": {"tags": [DRIVE_TAG]},
    "source": [
        "## Google Drive Login\n",
        "\n",
        "在实验开始时挂载一次 Google Drive。后面的保存 cell 会复用该挂载，",
        "因此长实验结束时不会再次弹出登录流程。",
    ],
}

DRIVE_CODE = {
    "cell_type": "code",
    "execution_count": None,
    "id": "google-drive-login",
    "metadata": {"tags": [DRIVE_TAG]},
    "outputs": [],
    "source": [
        "from pathlib import Path\n",
        "import sys\n",
        "\n",
        "DRIVE_RESULTS_ROOT = Path(\n",
        "    \"/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/\"\n",
        "    \"Synthetic_CoT_NiaH_Count/colab_results\"\n",
        ")\n",
        "DRIVE_MOUNTED = False\n",
        "\n",
        "def ensure_google_drive_mounted() -> bool:\n",
        "    global DRIVE_MOUNTED\n",
        "    if not (\"google.colab\" in sys.modules or Path(\"/content\").exists()):\n",
        "        print(\"Not in Colab; Google Drive mount skipped.\")\n",
        "        return False\n",
        "    from google.colab import drive\n",
        "    if not Path(\"/content/drive/MyDrive\").exists():\n",
        "        drive.mount(\"/content/drive\")\n",
        "    DRIVE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)\n",
        "    DRIVE_MOUNTED = True\n",
        "    print(\"Google Drive ready:\", DRIVE_RESULTS_ROOT)\n",
        "    return True\n",
        "\n",
        "ensure_google_drive_mounted()",
    ],
}

MOUNT_LINE = re.compile(
    r"^(?P<indent>\s*)drive\.mount\("
    r"(?P<quote>['\"])/content/drive(?P=quote)"
    r"(?:\s*,\s*force_remount\s*=\s*(?:True|False))?"
    r"\)\s*$"
)


def _replace_late_mounts(cell: dict) -> None:
    if cell.get("cell_type") != "code":
        return
    if DRIVE_TAG in cell.get("metadata", {}).get("tags", []):
        return

    source = "".join(cell.get("source", []))
    replaced = []
    for line in source.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        match = MOUNT_LINE.match(line.rstrip("\r\n"))
        if match:
            replaced.append(f"{match.group('indent')}ensure_google_drive_mounted(){newline}")
        else:
            replaced.append(line)
    cell["source"] = replaced


def _update_v7(notebook: dict) -> None:
    first = notebook["cells"][0]
    source = "".join(first.get("source", []))
    source = source.replace(
        "- length 1024：non-thinking + thinking；\n"
        "- length 2048：non-thinking + thinking。\n\n"
        "Main 一共训练四个模型。",
        "- length 2048：non-thinking + thinking。\n\n"
        "Main 一共训练两个模型。",
    )
    checkpoint_note = (
        "\n训练期间每 2000 steps 保存一次可恢复 checkpoint，并立即同步到 "
        "Google Drive；重新连接后会从 Drive 上最新 checkpoint 续训。\n"
    )
    if "每 2000 steps 保存一次可恢复 checkpoint" not in source:
        source = source.rstrip() + checkpoint_note
    first["source"] = source.splitlines(keepends=True)

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        old_call = (
            'combined = run_sweep("v7", PRESET, OUT_ROOT, '
            "skip_completed=SKIP_COMPLETED, device=DEVICE)"
        )
        if old_call not in code:
            continue
        new_call = (
            'LIVE_CHECKPOINT_ROOT = DRIVE_RESULTS_ROOT / "v7_live_checkpoints" '
            "if DRIVE_MOUNTED else None\n"
            "if LIVE_CHECKPOINT_ROOT is not None:\n"
            "    LIVE_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "display(Markdown(f\"**Live checkpoint root:** `{LIVE_CHECKPOINT_ROOT}`\"))\n\n"
            'combined = run_sweep("v7", PRESET, OUT_ROOT, '
            "skip_completed=SKIP_COMPLETED, device=DEVICE, "
            "checkpoint_sync_root=LIVE_CHECKPOINT_ROOT)"
        )
        cell["source"] = code.replace(old_call, new_call).splitlines(keepends=True)

    # The saved outputs still showed the removed 1024 setting and an interrupted
    # four-model run. A clean notebook prevents those stale results being read as
    # the current configuration.
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def _preferred_indent(path: Path) -> int:
    repo_root = NOTEBOOK_DIR.parent
    relative = path.resolve().relative_to(repo_root).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    reference = result.stdout if result.returncode == 0 else path.read_text(encoding="utf-8")
    for line in reference.splitlines()[:10]:
        if line.lstrip().startswith('"cells"'):
            return len(line) - len(line.lstrip())
    return 2


def update_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cells[:] = [
        cell
        for cell in cells
        if DRIVE_TAG not in cell.get("metadata", {}).get("tags", [])
        and "def ensure_google_drive_mounted" not in "".join(cell.get("source", []))
    ]

    first_code = next(i for i, cell in enumerate(cells) if cell.get("cell_type") == "code")
    cells[first_code:first_code] = [DRIVE_MARKDOWN.copy(), DRIVE_CODE.copy()]

    for cell in cells:
        _replace_late_mounts(cell)

    if path.name == "Trace_Count_v7_Colab.ipynb":
        _update_v7(notebook)

    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=_preferred_indent(path)) + "\n",
        encoding="utf-8",
    )


def verify_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    tagged = [
        i
        for i, cell in enumerate(cells)
        if DRIVE_TAG in cell.get("metadata", {}).get("tags", [])
    ]
    raw_mounts = [
        i
        for i, cell in enumerate(cells)
        if "drive.mount(" in "".join(cell.get("source", []))
    ]
    helper_calls = [
        i
        for i, cell in enumerate(cells)
        if "ensure_google_drive_mounted()" in "".join(cell.get("source", []))
    ]
    if (
        len(tagged) != 2
        or len(raw_mounts) != 1
        or len(helper_calls) < 2
        or max(tagged) > 5
    ):
        raise ValueError(
            f"Invalid Drive setup in {path.name}: tagged={tagged}, "
            f"raw_mounts={raw_mounts}, helper_calls={helper_calls}"
        )


def main() -> None:
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    for path in notebooks:
        update_notebook(path)
        verify_notebook(path)
        print(f"updated and verified {path.relative_to(NOTEBOOK_DIR.parent)}")


if __name__ == "__main__":
    main()
