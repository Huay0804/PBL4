"""Fresh-clone smoke test for the segmentation pipeline.

Catches the failure modes that have bitten this repo before:

* an *untracked* helper module (e.g. ``scripts/protocol_utils.py``) — a fresh
  clone would silently miss the file and crash on import;
* a tracked module on HEAD missing a symbol that other modules import (e.g.
  ``ce_dice_boundary_loss`` in ``helper_functions.py``);
* a build_model branch that references a model class not exported from the
  ``segmentation_models`` package;
* a constructor change that breaks at the protocol resolution.

Run locally with ``./venv/bin/python scripts/smoke_test.py`` (~10 s with TF
already installed). Also runs in CI on every push to main via
``.github/workflows/smoke-test.yml``.

The test is intentionally lightweight: it byte-compiles the pipeline, imports
every project-local module the entry points use, runs ``--help`` on each entry
point in a subprocess (which forces the full real import chain), and builds
each segmentation model at a 64x128 input to verify the constructors run end
to end. No data is read; no real training step is taken.
"""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

ENTRY_POINTS = (
    "scripts/train.py",
    "scripts/train_segmentation_cv.py",
    "scripts/evaluate_final.py",
)

PROJECT_LOCAL_MODULES = (
    "project_presets",
    "protocol_utils",
    "helper_functions",
)

SEG_MODEL_IMPORTS = (
    "ModifiedNestnet",
    "ICPRModifiedUnet",
    "ICPRUnet",
    "TransUNet",
)

# Used symbols from helper_functions — if any of these vanish, train.py crashes
# at import. Smoke test will refuse to pass before they vanish.
HELPER_REQUIRED = (
    "MeanIoUMetric",
    "dice_coef",
    "dice_coef_loss",
    "bce_dice_loss",
    "ce_dice_loss",
    "ce_dice_boundary_loss",
    "multiclass_dice_loss",
    "mean_iou",
    "iou_score",
)


def _print_stage(name):
    print(f"\n=== {name} ===", flush=True)


def stage_compileall():
    _print_stage("compileall scripts + src")
    subprocess.check_call([
        sys.executable, "-m", "compileall", "-q",
        str(SCRIPTS_DIR), str(SRC_DIR / "segmentation_models"),
    ])
    print("OK")


def stage_imports():
    _print_stage("project-local imports")
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(SCRIPTS_DIR))
    for mod in PROJECT_LOCAL_MODULES:
        __import__(mod)
        print(f"OK  import {mod}")

    import helper_functions  # noqa: E402
    missing = [s for s in HELPER_REQUIRED if not hasattr(helper_functions, s)]
    if missing:
        raise SystemExit(
            f"helper_functions missing required symbols: {missing}.\n"
            "train.py imports these directly; a fresh clone would crash at import."
        )
    print(f"OK  helper_functions has all {len(HELPER_REQUIRED)} required symbols")

    seg = __import__("segmentation_models", fromlist=list(SEG_MODEL_IMPORTS))
    missing = [n for n in SEG_MODEL_IMPORTS if not hasattr(seg, n)]
    if missing:
        raise SystemExit(
            f"segmentation_models missing exports: {missing}.\n"
            "Each is referenced by scripts/train.py's build_model branches."
        )
    print(f"OK  segmentation_models exports {list(SEG_MODEL_IMPORTS)}")


def stage_help_for_entrypoints():
    _print_stage("--help on every entry point (subprocess)")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only; just exercises import
    for script in ENTRY_POINTS:
        path = PROJECT_ROOT / script
        cp = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if cp.returncode != 0:
            raise SystemExit(
                f"{script} --help failed (exit {cp.returncode})\n"
                f"--- stderr ---\n{cp.stderr}"
            )
        print(f"OK  {script} --help")


def stage_build_models():
    _print_stage("build each segmentation model at 64x128 (constructor check)")
    # Late import so any TF complaint surfaces with a clear message above.
    from segmentation_models import (  # noqa: E402
        ModifiedNestnet,
        ICPRModifiedUnet,
        ICPRUnet,
        TransUNet,
    )
    classes = 33
    cases = [
        ("TransUNet", lambda: TransUNet(
            input_shape=(64, 128, 3), classes=classes, activation="softmax",
        )),
        ("ICPRUnet", lambda: ICPRUnet(
            input_shape=(64, 128, 3), classes=classes, activation="softmax",
        )),
        ("ModifiedNestnet", lambda: ModifiedNestnet(
            input_shape=(64, 128, 3), classes=classes, activation="softmax",
            bb_channels=classes,
        )),
        ("ICPRModifiedUnet", lambda: ICPRModifiedUnet(
            input_shape=(64, 128, 3), classes=classes, activation="softmax",
            bb_channels=classes,
        )),
    ]
    for name, factory in cases:
        model = factory()
        params = model.count_params()
        print(f"OK  {name:<18s} params={params:>12,d}")


def main():
    print(f"PYTHON: {sys.version.split()[0]}", flush=True)
    print(f"REPO:   {PROJECT_ROOT}", flush=True)

    stage_compileall()
    stage_imports()
    stage_help_for_entrypoints()
    stage_build_models()

    print("\nSMOKE_TEST_PASSED", flush=True)


if __name__ == "__main__":
    main()
