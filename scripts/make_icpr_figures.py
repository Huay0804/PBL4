import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_MODELS = ["icpr_unet-resnet18", "icpr_munet-resnet18"]
UPPER_JAW_CLASS_IDS = list(range(1, 17))
LOWER_JAW_CLASS_IDS = list(range(17, 33))
# Dataset class ids are ordered 1..32 clockwise starting from the image's
# top-left tooth. These labels convert that order into FDI notation.
UPPER_JAW_LABELS = [
    "T18",
    "T17",
    "T16",
    "T15",
    "T14",
    "T13",
    "T12",
    "T11",
    "T21",
    "T22",
    "T23",
    "T24",
    "T25",
    "T26",
    "T27",
    "T28",
]
LOWER_JAW_LABELS = [
    "T38",
    "T37",
    "T36",
    "T35",
    "T34",
    "T33",
    "T32",
    "T31",
    "T41",
    "T42",
    "T43",
    "T44",
    "T45",
    "T46",
    "T47",
    "T48",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ICPR-style quantitative plots from runs/cv summaries."
    )
    p.add_argument("--cv-dir", type=Path, default=Path("runs/cv"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/cv/figures"))
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model run directory names inside each fold (e.g., icpr_unet-resnet18).",
    )
    p.add_argument(
        "--summary-per-position",
        type=Path,
        default=Path("runs/cv/summary_per_position_test.json"),
    )
    p.add_argument(
        "--summary-tooth-type",
        type=Path,
        default=Path("runs/cv/summary_tooth_type_test.json"),
    )
    p.add_argument(
        "--mask-rcnn-summary",
        type=Path,
        default=Path("runs/mask_rcnn/cv_summary.json"),
    )
    return p.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text())


def _model_label(model_dir: str) -> str:
    # Keep labels short and paper-like.
    if model_dir.startswith("icpr_unet"):
        return "U-Net"
    if model_dir.startswith("icpr_munet"):
        return "Mod-U-Net"
    return model_dir


def _collect_overall_test_metrics(cv_dir: Path, models: list[str]):
    overall = {}
    for model in models:
        fold_ious = []
        fold_dices = []
        for i in range(4):
            path = cv_dir / f"fold_{i}" / model / "per_class_metrics_test.json"
            if not path.exists():
                continue
            rows = _load_json(path)
            rows = [r for r in rows if int(r["class_id"]) != 0]
            if not rows:
                continue
            fold_ious.append(float(np.mean([float(r["iou"]) for r in rows])))
            fold_dices.append(float(np.mean([float(r["dice"]) for r in rows])))
        if not fold_ious:
            continue
        overall[model] = {
            "iou_mean": float(np.mean(fold_ious)),
            "iou_std": float(np.std(fold_ious)),
            "dice_mean": float(np.mean(fold_dices)),
            "dice_std": float(np.std(fold_dices)),
            "folds": len(fold_ious),
        }
    return overall


def plot_fig1_mask_rcnn_map(mask_summary: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = mask_summary.get("folds", [])
    if not folds:
        return None

    x = [int(row["fold"]) for row in folds]
    y = [float(row["mAP@0.5"]) for row in folds]
    agg = mask_summary.get("aggregate", {})
    mean_map = agg.get("mean_mAP@0.5")
    std_map = agg.get("std_mAP@0.5")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(x, y, width=0.6, color="#4C72B0")
    if mean_map is not None:
        ax.axhline(float(mean_map), color="#DD8452", linestyle="--", linewidth=1.2, label="CV mean")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Fold")
    ax.set_ylabel("mAP@0.5")
    title = "Mask R-CNN mAP@0.5 per fold"
    if mean_map is not None and std_map is not None:
        title += f" (mean={float(mean_map):.4f}, std={float(std_map):.4f})"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    if mean_map is not None:
        ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "fig1_mask_rcnn_map_cv.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_fig2_overall_test(overall: dict, models: list[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [_model_label(m) for m in models if m in overall]
    model_keys = [m for m in models if m in overall]
    if not model_keys:
        return None

    iou_means = [overall[m]["iou_mean"] for m in model_keys]
    iou_stds = [overall[m]["iou_std"] for m in model_keys]
    dice_means = [overall[m]["dice_mean"] for m in model_keys]
    dice_stds = [overall[m]["dice_std"] for m in model_keys]

    x = np.arange(len(model_keys))
    w = 0.36

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
    ax.bar(x - w / 2, iou_means, width=w, yerr=iou_stds, capsize=4, label="IoU")
    ax.bar(x + w / 2, dice_means, width=w, yerr=dice_stds, capsize=4, label="Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Overall test IoU/Dice (mean ± std across folds)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "fig2_overall_test_iou_dice.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _extract_series(by_class_id: dict[int, dict], class_ids: list[int], key: str):
    return [float(by_class_id.get(class_id, {}).get(key, 0.0)) for class_id in class_ids]


def _radar_plot(
    ax,
    model_series: list[tuple[str, list[float]]],
    labels: list[str],
    radial_max: float,
    panel_label: str,
):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.0, radial_max)
    ax.grid(alpha=0.35)
    if radial_max <= 0.2:
        r_ticks = np.linspace(0.0, radial_max, 5)
    else:
        r_ticks = np.linspace(0.0, radial_max, 6)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{v:.2f}".rstrip("0").rstrip(".") for v in r_ticks], fontsize=7)
    ax.set_title("Teeth", fontsize=9, pad=10)

    for label, values in model_series:
        values_closed = np.concatenate([values, [values[0]]])
        ax.plot(angles_closed, values_closed, linewidth=1.4, label=label)
        ax.fill(angles_closed, values_closed, alpha=0.12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=7, frameon=True)
    ax.text(0.5, -0.28, panel_label, transform=ax.transAxes, ha="center", va="center", fontsize=11)


def plot_fig3_per_tooth_dice(summary_per_position: dict, models: list[str], out_dir: Path):
    # ICPR Fig. 3 style:
    # (a) upper jaw Dice means, (b) upper jaw Dice std,
    # (c) lower jaw Dice means, (d) lower jaw Dice std.
    out_dir.mkdir(parents=True, exist_ok=True)

    available = []
    for model in models:
        rows = summary_per_position.get(model)
        if not rows:
            continue
        by_id = {int(r["class_id"]): r for r in rows}
        available.append((_model_label(model), by_id))
    if not available:
        return None

    upper_mean_series = [
        (label, _extract_series(by_id, UPPER_JAW_CLASS_IDS, "dice_mean"))
        for label, by_id in available
    ]
    upper_std_series = [
        (label, _extract_series(by_id, UPPER_JAW_CLASS_IDS, "dice_std"))
        for label, by_id in available
    ]
    lower_mean_series = [
        (label, _extract_series(by_id, LOWER_JAW_CLASS_IDS, "dice_mean"))
        for label, by_id in available
    ]
    lower_std_series = [
        (label, _extract_series(by_id, LOWER_JAW_CLASS_IDS, "dice_std"))
        for label, by_id in available
    ]

    std_values = [v for _, vals in (upper_std_series + lower_std_series) for v in vals]
    std_max = max(std_values) if std_values else 0.1
    std_max = max(0.1, min(1.0, np.ceil(std_max / 0.05) * 0.05))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        subplot_kw={"projection": "polar"},
    )
    _radar_plot(axes[0, 0], upper_mean_series, UPPER_JAW_LABELS, 1.0, "(a)")
    _radar_plot(axes[0, 1], upper_std_series, UPPER_JAW_LABELS, std_max, "(b)")
    _radar_plot(axes[1, 0], lower_mean_series, LOWER_JAW_LABELS, 1.0, "(c)")
    _radar_plot(axes[1, 1], lower_std_series, LOWER_JAW_LABELS, std_max, "(d)")

    fig.tight_layout()
    out_path = out_dir / "fig3_per_tooth_dice_test.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_fig4_tooth_type_dice(summary_tooth_type: dict, models: list[str], out_dir: Path):
    # ICPR Fig. 4: Dice by tooth type on test set, with error bars.
    out_dir.mkdir(parents=True, exist_ok=True)

    tooth_types = ["incisor", "canine", "premolar", "molar"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(tooth_types))
    width = 0.35 if len(models) <= 2 else 0.8 / max(1, len(models))

    for i, model in enumerate(models):
        rows = summary_tooth_type.get(model)
        if not rows:
            continue
        by_group = {r["group"]: r for r in rows}
        means = [float(by_group[t]["dice_mean"]) for t in tooth_types]
        stds = [float(by_group[t]["dice_std"]) for t in tooth_types]
        offset = (i - (len(models) - 1) / 2.0) * width
        ax.bar(x + offset, means, width=width, label=_model_label(model), yerr=stds, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(tooth_types)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Dice")
    ax.set_title("Dice by tooth type (test set): mean ± std across folds")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path = out_dir / "fig4_tooth_type_dice_test.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()

    if not args.summary_per_position.exists():
        raise SystemExit(
            f"Missing {args.summary_per_position}. Run `python scripts/summarize_icpr_metrics.py` first."
        )
    if not args.summary_tooth_type.exists():
        raise SystemExit(
            f"Missing {args.summary_tooth_type}. Run `python scripts/summarize_icpr_metrics.py` first."
        )

    summary_per_position = _load_json(args.summary_per_position)
    summary_tooth_type = _load_json(args.summary_tooth_type)
    mask_rcnn_summary = (
        _load_json(args.mask_rcnn_summary) if args.mask_rcnn_summary.exists() else {}
    )
    overall_test = _collect_overall_test_metrics(args.cv_dir, args.models)

    fig1 = plot_fig1_mask_rcnn_map(mask_rcnn_summary, args.out_dir)
    fig2 = plot_fig2_overall_test(overall_test, args.models, args.out_dir)
    fig3 = plot_fig3_per_tooth_dice(summary_per_position, args.models, args.out_dir)
    fig4 = plot_fig4_tooth_type_dice(summary_tooth_type, args.models, args.out_dir)

    if fig1 is not None:
        print("Wrote:", fig1)
    if fig2 is not None:
        print("Wrote:", fig2)
    print("Wrote:", fig3)
    print("Wrote:", fig4)


if __name__ == "__main__":
    main()
