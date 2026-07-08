from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
DEFAULT_REPO_DIR = Path("/content/Synthetic_CoT_NiaH_Count")


def in_colab_like_runtime() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists()


def ensure_repo(repo_url: str = DEFAULT_REPO_URL, repo_dir: Path = DEFAULT_REPO_DIR, pull: bool = True) -> Path:
    if not in_colab_like_runtime():
        return Path.cwd()
    if repo_dir.exists():
        os.chdir(repo_dir)
        if pull and (repo_dir / ".git").exists():
            subprocess.run(["git", "pull"], check=False)
    else:
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
        os.chdir(repo_dir)
    return Path.cwd()


def repair_science_stack_if_needed(repair: bool = True) -> None:
    try:
        import numpy as np
        import pandas as pd
        import scipy

        print(f"numpy={np.__version__} pandas={pd.__version__} scipy={scipy.__version__}")
        return
    except (ImportError, ValueError, AttributeError) as exc:
        msg = str(exc)
        abi_markers = ("numpy.dtype size changed", "_ARRAY_API", "compiled using NumPy 1.x", "multiarray failed")
        if not any(marker in msg for marker in abi_markers):
            raise
        print("Detected NumPy/scientific-stack ABI mismatch:", msg)
        if not repair:
            raise RuntimeError("Set REPAIR_NUMPY_ABI=True, rerun setup, then restart the kernel.") from exc
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--force-reinstall",
                "numpy<2",
                "pandas",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "seaborn",
            ],
            check=True,
        )
        raise RuntimeError("Repaired NumPy ABI packages. Restart the kernel, then rerun setup.") from exc


def install_minimal_deps() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers>=4.41",
            "tqdm",
            "pandas",
            "matplotlib",
            "seaborn",
            "scikit-learn",
        ],
        check=True,
    )


def import_check(modules: Iterable[str]) -> None:
    for module in modules:
        importlib.import_module(module)


def setup_colab(
    *,
    repo_url: str = DEFAULT_REPO_URL,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    pull: bool = True,
    repair_numpy_abi: bool = True,
    install_deps: bool = False,
    install_editable: bool = False,
    import_modules: Iterable[str] = ("torch", "transformers", "pandas", "sklearn", "matplotlib", "seaborn"),
) -> Path:
    cwd = ensure_repo(repo_url=repo_url, repo_dir=Path(repo_dir), pull=pull)
    if install_deps:
        install_minimal_deps()
    repair_science_stack_if_needed(repair=repair_numpy_abi)
    if install_editable:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)
    import_check(import_modules)
    print("cwd =", cwd)
    print("Dependency import check passed.")
    return cwd
