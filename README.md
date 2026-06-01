# PBL4 Teeth Segmentation

This repository reproduces a two-stage teeth segmentation pipeline on panoramic
dental X-ray images:

1. `Mask R-CNN` detects teeth and exports class-wise bounding-box prior maps.
2. `ICPR U-Net` / `ICPR Modified U-Net` perform semantic segmentation, with the
   modified model consuming the exported prior maps.

An optional vendored `YOLOX` detector is also available under `src/yolox/` as a
box-only alternative to Mask R-CNN for prior generation.

The model architecture and overall pipeline follow the reference ICPR paper.
The code in this repository is the project implementation of that pipeline.

## Credits and References

- Reference pipeline paper:
  [Automatic tooth segmentation on panoramic X-rays using deep neural networks (ICPR 2022)](https://www.polytech.univ-nantes.fr/autrusseau-f/Papers/ICPR2022_Odon.pdf)
- TF2 Mask R-CNN code source used in this project:
  [z-mahmud22/Mask-RCNN_TF2.14.0](https://github.com/z-mahmud22/Mask-RCNN_TF2.14.0)
- Original Mask R-CNN implementation that the TF2 port is based on:
  [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- YOLOX detector code source used in this project:
  [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

The vendored `src/mrcnn_tf2/` code in this repository is credited to the
TensorFlow 2 port above and to the original Matterport `Mask_RCNN` project,
whose source headers and license notices are still preserved locally under the
MIT license.

## Environments

Two Python environments are used:

- `venv/` for segmentation training, evaluation, summaries, and figures
- `src/mrcnn_tf2/mrcnn-tf2-venv/` for Mask R-CNN training and detection
- optionally, a YOLOX-capable environment for `scripts/train_yolox.py`

### Tested OS

The current local setup used for this project is:

- Ubuntu 24.04.3 LTS
- Linux kernel `6.17.0-14-generic`
- `x86_64` architecture
- Python `3.12` for the segmentation `venv`

### Segmentation venv setup

Create the main project virtual environment for segmentation with Python 3.12:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Quick check:

```bash
source venv/bin/activate
python -c "import tensorflow as tf; import keras; print(tf.__version__, keras.__version__)"
```

Use this environment for:

- `scripts/train_segmentation_cv.py`
- `scripts/train.py`
- `scripts/evaluate_final.py`
- `scripts/summarize_icpr_metrics.py`
- `scripts/make_icpr_figures.py`
- optionally, `scripts/train_yolox.py` if PyTorch + YOLOX deps are installed

### Current segmentation package versions

The required segmentation environment is pinned to:

- `tensorflow==2.20.0`
- `keras==3.13.0`
- `numpy==2.0.2`
- `scipy==1.15.1`
- `scikit-image==0.25.2`
- `pillow==12.1.0`
- `h5py==3.12.1`

### Optional augmentation dependency

`imgaug` has been removed from the tracked requirements because the current
ICPR workflow in this repository does not enable the optional augmentation path
that depends on it. This applies to both:

- the segmentation environment in `venv/`
- the vendored TF2 Mask R-CNN environment in `src/mrcnn_tf2/mrcnn-tf2-venv/`

If augmentation based on the upstream Mask R-CNN hooks is needed later,
`imgaug` can still be installed manually as an extra dependency.

### Optional YOLOX environment

If you want to train or run the vendored YOLOX detector locally, create a
PyTorch-capable environment and install YOLOX from source:

```bash
python3.12 -m venv src/yolox/yolox-venv
source src/yolox/yolox-venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r src/yolox/requirements-pbl4.txt
pip install -e src/yolox
```

Use this environment for:

- `scripts/train_yolox.py`

## Repository Layout

- `scripts/train_mask_rcnn.py`: Mask R-CNN train/detect entrypoint
- `scripts/train_yolox.py`: YOLOX train/detect entrypoint
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
- `src/yolox/`: vendored YOLOX detector codebase

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

This writes priors under:

```text
data/bb_maps/mask_rcnn/{test,folds/fold_<k>/{train,val}}/bb_maps
```

### 4b. Train YOLOX on a single fold

```bash
src/yolox/yolox-venv/bin/python scripts/train_yolox.py --mode train --fold 0
```

### 4c. Export YOLOX BB maps

Export train/val priors for fold `0` and include the fixed test split:

```bash
src/yolox/yolox-venv/bin/python scripts/train_yolox.py --mode detect --source-fold 0 --target-fold 0 --include-test
```

This writes YOLOX priors under:

```text
data/bb_maps/yolox/{test,folds/fold_<k>/{train,val}}/bb_maps
```

To train BB-gated segmentation with a specific BB source:

```bash
venv/bin/python scripts/train_segmentation_cv.py --model icpr_munet --fold 0 --bb-source yolox --eval-test
venv/bin/python scripts/train_segmentation_cv.py --model icpr_munet --fold 0 --bb-source mask_rcnn --eval-test
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
venv/bin/python scripts/evaluate_final.py --model icpr_munet --cv-fold 0 --bb-source yolox
venv/bin/python scripts/evaluate_final.py --model icpr_munet --cv-fold 0 --bb-source mask_rcnn
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
