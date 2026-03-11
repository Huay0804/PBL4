# PBL4 Teeth Segmentation

This repository reproduces a two-stage teeth segmentation pipeline on panoramic
dental X-ray images:

1. `Mask R-CNN` detects teeth and exports class-wise bounding-box prior maps.
2. `ICPR U-Net` / `ICPR Modified U-Net` perform semantic segmentation, with the
   modified model consuming the exported prior maps.

The model architecture and overall pipeline follow the reference ICPR paper.
The code in this repository is the project implementation of that pipeline.

## Environments

Two Python environments are used:

- `venv/` for segmentation training, evaluation, summaries, and figures
- `src/mrcnn_tf2/mrcnn-tf2-venv/` for Mask R-CNN training and detection

## Repository Layout

- `scripts/train_mask_rcnn.py`: Mask R-CNN train/detect entrypoint
- `scripts/train_segmentation_cv.py`: segmentation training over a selected CV fold
- `scripts/train.py`: low-level segmentation training/evaluation engine
- `scripts/evaluate_final.py`: fixed test-set evaluation for segmentation models
- `scripts/evaluate_mask_rcnn_icpr.py`: ICPR-style Mask R-CNN test evaluation
- `scripts/summarize_icpr_metrics.py`: aggregate per-fold ICPR metrics
- `scripts/make_icpr_figures.py`: render report figures from saved metrics
- `scripts/prepare_data.py`: raw annotation to semantic-mask preparation
- `scripts/split_dataset.py`: fixed test split plus 4-fold CV split generation
- `src/segmentation_models/icpr_unet/model.py`: ICPR U-Net and Modified U-Net
- `src/mrcnn_tf2/`: TF2-compatible Mask R-CNN codebase

## Typical Workflow

### 1. Prepare data

Run only when rebuilding processed masks from raw polygon annotations.

```bash
venv/bin/python scripts/prepare_data.py
```

### 2. Build the fixed test split and 4 CV folds

Run only when regenerating the split folders.

```bash
venv/bin/python scripts/split_dataset.py
```

### 3. Train Mask R-CNN on a single fold

```bash
src/mrcnn_tf2/mrcnn-tf2-venv/bin/python scripts/train_mask_rcnn.py --mode train --fold 0
```

### 4. Export BB maps with a trained Mask R-CNN fold

Export to the fixed test split:

```bash
src/mrcnn_tf2/mrcnn-tf2-venv/bin/python scripts/train_mask_rcnn.py --mode detect --source-fold 0
```

Export to a target CV fold (`train/val` inside that fold):

```bash
src/mrcnn_tf2/mrcnn-tf2-venv/bin/python scripts/train_mask_rcnn.py --mode detect --source-fold 0 --target-fold 0
```

### 5. Train a segmentation model on one CV fold

```bash
venv/bin/python scripts/train_segmentation_cv.py --model icpr_unet --fold 0
venv/bin/python scripts/train_segmentation_cv.py --model icpr_munet --fold 0
```

### 6. Evaluate a trained segmentation fold on the fixed test set

```bash
venv/bin/python scripts/evaluate_final.py --model icpr_unet --cv-fold 0
venv/bin/python scripts/evaluate_final.py --model icpr_munet --cv-fold 0
```

### 7. Build aggregated summaries and figures

```bash
venv/bin/python scripts/summarize_icpr_metrics.py
venv/bin/python scripts/make_icpr_figures.py
```

### 8. Run ICPR-style Mask R-CNN test evaluation

```bash
src/mrcnn_tf2/mrcnn-tf2-venv/bin/python scripts/evaluate_mask_rcnn_icpr.py --subset test
```

## Notes

- The repository ignores `data/`, `runs/`, and other large artifacts by default.
- `.docx` reports and temporary notebook/output folders are also ignored.
- Use the fixed `data/splits/test` set for final comparisons and the `fold_*`
  directories for 4-fold cross-validation experiments.
