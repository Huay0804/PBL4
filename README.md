# PBL4 Teeth Segmentation

This repository contains the code used for a comparative study of tooth
segmentation models on panoramic dental X-ray images. The intended way to
reproduce the experiments is through the Kaggle notebooks in the repository,
not by setting up the full stack locally.

The notebooks clone this repository inside a Kaggle session, link the mounted
Kaggle datasets to the paths expected by the scripts, train the selected model
over 4 cross-validation folds, evaluate each fold on the fixed test set, and
package the checkpoints and metrics for download.

## Kaggle-First Workflow

Use Kaggle for training and evaluation. Local execution is not the recommended
entrypoint because the project combines TensorFlow/Keras segmentation models,
YOLO/YOLOX tooling, Mask R-CNN prior maps, large datasets, and GPU-specific
memory settings.

Each notebook is self-contained and follows the same structure:

1. Configure repository and dataset paths.
2. Clone or update this repository from GitHub.
3. Link mounted Kaggle datasets into `data/splits` and, when needed,
   `data/bb_maps`.
4. Train all 4 folds.
5. Evaluate each fold on the fixed test set.
6. Print summary metrics.
7. Zip model checkpoints and result files into `/kaggle/working`.

Interrupted runs can be resumed. The notebooks skip folds that already contain
the expected best checkpoint, so a Kaggle session can continue from uploaded or
previously generated results.

## Notebooks

| Notebook | Purpose | Required Kaggle inputs |
| --- | --- | --- |
| `kaggle_icpr_unet.ipynb` | Image-only ICPR U-Net baseline | Split dataset only |
| `kaggle_icpr_munet.ipynb` | ICPR Modified U-Net with Mask R-CNN bounding-box priors | Split dataset + Mask R-CNN BB maps |
| `kaggle_mod_nestnet.ipynb` | Modified NestNet / UNet++ with YOLOX bounding-box priors | Split dataset + YOLOX BB maps |
| `kaggle_transunet.ipynb` | Image-only TransUNet baseline | Split dataset only |
| `kaggle_yolo_seg.ipynb` | YOLO instance-segmentation baselines (`yolo11`, `yolo26`) | Split dataset only |

Recommended execution order for reproducing the comparison:

1. Run the image-only baselines:
   - `kaggle_icpr_unet.ipynb`
   - `kaggle_transunet.ipynb`
2. Run the prior-gated models:
   - `kaggle_icpr_munet.ipynb`
   - `kaggle_mod_nestnet.ipynb`
3. Run the detector-segmentation baselines:
   - `kaggle_yolo_seg.ipynb`
4. Download each notebook's result zip from `/kaggle/working`.

## Required Kaggle Datasets

The notebooks expect the data to be mounted through Kaggle's **Add data**
panel. The exact dataset slugs can be different; the notebooks search under
`/kaggle/input` and create symlinks automatically.

### 1. Split Dataset

All notebooks need a prepared split dataset containing the fixed test split and
4 cross-validation folds. Use the Kaggle dataset:

```text
https://www.kaggle.com/datasets/hieuminhhale/pbl4-splits
```

It should expose a `splits` directory with this general layout:

```text
splits/
  class_map.txt
  test/
    images/
    masks_semantic/
  folds/
    fold_0/
      train/
      val/
    fold_1/
      train/
      val/
    fold_2/
      train/
      val/
    fold_3/
      train/
      val/
```

Inside each `train`, `val`, or `test` split, the scripts expect image files and
semantic masks in the same format used during this project.

### 2. Bounding-Box Prior Maps

Only the prior-gated segmentation notebooks need bounding-box prior maps.

For `kaggle_icpr_munet.ipynb`, mount a dataset containing Mask R-CNN priors:

```text
bb_maps/
  mask_rcnn/
    test/
      bb_maps/
    folds/
      fold_0/
        train/
          bb_maps/
        val/
          bb_maps/
      ...
```

For `kaggle_mod_nestnet.ipynb`, mount a dataset containing YOLOX priors:

```text
bb_maps/
  yolox/
    test/
      bb_maps/
    folds/
      fold_0/
        train/
          bb_maps/
        val/
          bb_maps/
      ...
```

The notebooks link these folders into the repository as `data/bb_maps/...`.

## How to Run on Kaggle

1. Create a Kaggle notebook with GPU enabled.
2. Add the required Kaggle datasets:
   - the prepared split dataset,
   - plus BB-prior datasets only for prior-gated models.
3. Upload or copy one of the notebooks from this repository.
4. In the first configuration cell, check:
   - `REPO_URL`,
   - `REPO_DIR`,
   - any explicit dataset path if the notebook exposes one.
5. Run all cells from top to bottom.
6. Download the generated zip files from `/kaggle/working`.

If the GitHub repository is private, replace `REPO_URL` in the notebook with an
authenticated URL or make the repository accessible to the Kaggle session.

## Outputs

Dense segmentation notebooks package two zip files:

```text
<model>_models.zip
<model>_results.zip
```

The model zip contains the best fold checkpoints, usually `best.keras`. The
results zip contains metrics and summaries such as:

```text
test_summary.json
test_metrics.json
per_class_metrics_test.json
per_position_metrics_test.json
per_tooth_type_metrics_test.json
```

The YOLO-seg notebook similarly writes:

```text
yolo_seg_models.zip
yolo_seg_results.zip
```

The YOLO model zip contains `best.pt` checkpoints. The results zip contains the
same fixed-test metric format used by the dense segmentation models, plus YOLO
summary files and plots when available.

## Repository Layout

The notebooks are the public entrypoints. The Python scripts and packages are
implementation details called by those notebooks.

```text
kaggle_icpr_unet.ipynb       # ICPR U-Net baseline
kaggle_icpr_munet.ipynb      # Modified U-Net with Mask R-CNN priors
kaggle_mod_nestnet.ipynb     # Modified NestNet with YOLOX priors
kaggle_transunet.ipynb       # TransUNet baseline
kaggle_yolo_seg.ipynb        # YOLO11/YOLO26 segmentation baselines

scripts/
  train_segmentation_cv.py   # CV wrapper for dense segmentation models
  train.py                   # Keras segmentation training/evaluation engine
  evaluate_final.py          # Fixed-test evaluation
  evaluate_yolo_seg.py       # YOLO-seg fixed-test evaluation
  prepare_yolo_seg_data.py   # YOLO-seg dataset materialization
  project_presets.py         # Shared training protocol presets

src/segmentation_models/     # U-Net, Modified U-Net, NestNet, TransUNet code
src/mrcnn_tf2/                # Vendored TF2 Mask R-CNN code
src/yolox/                    # Vendored YOLOX code
apps/tooth_charting_assistant # Optional demo application
```

Large artifacts are intentionally not committed. This includes:

```text
data/
runs/
outputs/
checkpoints/
*.keras
*.pt
*.docx
```

Keep generated results in Kaggle outputs or external storage, not in the Git
repository.

## Experiment Protocol

The notebooks use the same high-level comparison protocol:

- 4-fold cross-validation for training and validation.
- A fixed held-out test set for final fold evaluation.
- Best-checkpoint evaluation for each fold.
- Shared semantic label space of 33 classes.
- Dense segmentation metrics exported in the same JSON format across models.

Important model-specific notes:

- `icpr_unet` and `transunet` are image-only baselines.
- `icpr_munet` consumes Mask R-CNN bounding-box prior maps.
- `mod_nestnet` consumes YOLOX bounding-box prior maps and uses the configured
  deep-supervision head used in the project protocol.
- `yolo11` and `yolo26` are detector-segmentation baselines whose instance masks
  are rasterized into the same semantic evaluation format.

## Credits and References

- Reference pipeline paper:
  [Automatic tooth segmentation on panoramic X-rays using deep neural networks (ICPR 2022)](https://www.polytech.univ-nantes.fr/autrusseau-f/Papers/ICPR2022_Odon.pdf)
- TF2 Mask R-CNN code source used in this project:
  [z-mahmud22/Mask-RCNN_TF2.14.0](https://github.com/z-mahmud22/Mask-RCNN_TF2.14.0)
- Original Mask R-CNN implementation that the TF2 port is based on:
  [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- YOLOX detector code source used in this project:
  [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

The vendored `src/mrcnn_tf2/` code is credited to the TensorFlow 2 port above
and to the original Matterport `Mask_RCNN` project. Source headers and license
notices are preserved locally under the MIT license.
