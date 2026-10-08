# PBL4 Teeth Segmentation

## Overview

This project compares six models for 33-class semantic tooth segmentation in
panoramic dental X-rays. Training and evaluation are organized in Kaggle
notebooks, with four cross-validation folds and a fixed held-out test set.

## Models

| Model | Input | Notebook |
| --- | --- | --- |
| ICPR U-Net | X-ray | `kaggle_icpr_unet.ipynb` |
| ICPR Modified U-Net | X-ray + Mask R-CNN bounding-box priors | `kaggle_icpr_munet.ipynb` |
| Modified NestNet / UNet++ | X-ray + YOLOX bounding-box priors | `kaggle_mod_nestnet.ipynb` |
| TransUNet | X-ray | `kaggle_transunet.ipynb` |
| YOLO11-seg and YOLO26-seg | X-ray | `kaggle_yolo_seg.ipynb` |

Modified NestNet uses the third output head (zero-based index 2) for training
and evaluation. YOLO instance masks are converted to semantic masks for
evaluation. Mask R-CNN and YOLOX provide the bounding-box priors for the two
prior-guided models.

## Dataset

Add the [prepared dataset on Kaggle](https://www.kaggle.com/datasets/hieuminhhale/pbl4-splits)
to the notebook. It includes the image splits, semantic masks, and bounding-box
prior maps needed by all five notebooks.

The dataset contains 598 images:

- **110 images** form the fixed test set.
- **488 images** are divided into four cross-validation folds, each with
  **366 training** and **122 validation** images.

The notebooks locate `splits/class_map.txt` under `/kaggle/input` and link the
mounted data to the expected local paths. In the layout below, braces indicate
separate directories: `{img,masks_semantic}` means `img/` and `masks_semantic/`.

```text
splits/
  class_map.txt
  test/{img,masks_semantic}/
  folds/fold_0/{train,val}/{img,masks_semantic}/
  ...                         # folds 1–3
bb_maps/
  mask_rcnn/
    test/bb_maps/
    folds/fold_0/{train,val}/bb_maps/
    ...                       # folds 1–3
  yolox/                      # same layout as mask_rcnn
```

## Running the Experiments

1. Open one of the notebooks on Kaggle and enable a GPU.
2. Add the `pbl4-splits` dataset.
3. Check `REPO_URL`, `REPO_DIR`, and any dataset path overrides in the
   configuration cell. The repository must be accessible from the Kaggle session.
4. Run the notebook and download the output archives from `/kaggle/working`.

Each notebook sets up the data paths, trains across all four folds, and evaluates
each fold's best checkpoint on the fixed test set. Folds with an existing best
checkpoint are skipped during training, allowing previously saved outputs to be
reused.

## Checkpoints and Outputs

Download the [trained model checkpoints from Google Drive](https://drive.google.com/drive/folders/1DgqOt-Sio_r9KKLF5CJzK96l2EOXfv7O?usp=sharing).

Each fold exports test summaries and metrics by class, tooth position, and tooth
type.

Dense-model notebooks export `<model>_models.zip` (`.keras` checkpoints) and
`<model>_results.zip` (evaluation JSON files). The YOLO notebook exports
`yolo_seg_models.zip` (`best.pt` checkpoints) and `yolo_seg_results.zip`.

## References

- [Automatic tooth segmentation on panoramic X-rays using deep neural networks (ICPR 2022)](https://www.polytech.univ-nantes.fr/autrusseau-f/Papers/ICPR2022_Odon.pdf) — reference pipeline.
- [Mask-RCNN_TF2.14.0](https://github.com/z-mahmud22/Mask-RCNN_TF2.14.0) — TensorFlow 2 port used in this project, based on [Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN). Source headers and MIT license notices are retained in `src/mrcnn_tf2/`.
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) — detector implementation used for YOLOX priors.
