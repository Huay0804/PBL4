#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from pathlib import Path

from yolox.data import COCODataset, TrainTransform, ValTransform
from yolox.exp import Exp as MyExp


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = _env_int("PBL4_YOLOX_NUM_CLASSES", 32)
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.environ.get("PBL4_YOLOX_EXPERIMENT_NAME", "yolox_teeth_s")
        self.output_dir = os.environ.get(
            "PBL4_YOLOX_OUTPUT_DIR",
            str(PROJECT_ROOT / "runs" / "yolox" / "fold_0"),
        )

        self.data_dir = os.environ.get(
            "PBL4_YOLOX_DATA_DIR",
            str(PROJECT_ROOT / "data" / "splits" / "folds" / "fold_0" / "yolox_coco"),
        )
        self.train_ann = os.environ.get("PBL4_YOLOX_TRAIN_ANN", "instances_train2017.json")
        self.val_ann = os.environ.get("PBL4_YOLOX_VAL_ANN", "instances_val2017.json")
        self.test_ann = os.environ.get("PBL4_YOLOX_TEST_ANN", "instances_test2017.json")
        self.train_name = os.environ.get("PBL4_YOLOX_TRAIN_NAME", "train2017")
        self.val_name = os.environ.get("PBL4_YOLOX_VAL_NAME", "val2017")
        self.test_name = os.environ.get("PBL4_YOLOX_TEST_NAME", "test2017")

        input_height = _env_int("PBL4_YOLOX_INPUT_H", 512)
        input_width = _env_int("PBL4_YOLOX_INPUT_W", 1024)
        self.input_size = (input_height, input_width)
        self.test_size = (input_height, input_width)
        self.multiscale_range = 0
        self.data_num_workers = _env_int("PBL4_YOLOX_NUM_WORKERS", 4)

        # Tooth IDs encode spatial positions, so left-right flips are invalid.
        self.flip_prob = 0.0
        self.hsv_prob = 0.0
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.enable_mixup = False
        self.degrees = 0.0
        self.translate = 0.05
        self.mosaic_scale = (1.0, 1.0)
        self.mixup_scale = (1.0, 1.0)
        self.shear = 0.0

        self.max_epoch = _env_int("PBL4_YOLOX_EPOCHS", 80)
        self.warmup_epochs = _env_int("PBL4_YOLOX_WARMUP_EPOCHS", 5)
        self.no_aug_epochs = _env_int("PBL4_YOLOX_NO_AUG_EPOCHS", 15)
        self.eval_interval = _env_int("PBL4_YOLOX_EVAL_INTERVAL", 1)
        self.basic_lr_per_img = _env_float("PBL4_YOLOX_BASIC_LR_PER_IMG", 0.001 / 64.0)
        self.test_conf = _env_float("PBL4_YOLOX_TEST_CONF", 0.01)
        self.nmsthre = _env_float("PBL4_YOLOX_NMS", 0.55)
        self.best_metric = "ap50"
        self.save_history_ckpt = False
        self.save_best_only = True

    def get_dataset(self, cache=False, cache_type="ram"):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.train_name,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann if testdev else self.val_ann,
            name=self.test_name if testdev else self.val_name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
