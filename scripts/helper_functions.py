import numpy as np
import tensorflow as tf


class MeanIoUMetric(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        return super().update_state(y_true, y_pred, sample_weight)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * bce - dice_coef(y_true, y_pred)


def multiclass_dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.int32)
    if y_true.shape.rank is not None and y_true.shape.rank == y_pred.shape.rank:
        y_true_one_hot = tf.cast(y_true, tf.float32)
    else:
        y_true_one_hot = tf.one_hot(y_true, tf.shape(y_pred)[-1], dtype=tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = (0, 1, 2)
    intersection = tf.reduce_sum(y_true_one_hot * y_pred, axis=axes)
    denom = tf.reduce_sum(y_true_one_hot + y_pred, axis=axes)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return tf.reduce_mean(dice)


def multiclass_dice_loss(y_true, y_pred):
    return 1.0 - multiclass_dice_coef(y_true, y_pred)


def ce_dice_loss(y_true, y_pred):
    if y_pred.shape.rank is not None and y_pred.shape[-1] == 1:
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        ce = tf.reduce_mean(ce)
        return ce + dice_coef_loss(y_true, y_pred)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    ce = tf.reduce_mean(ce)
    return ce + multiclass_dice_loss(y_true, y_pred)


def iou_score(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_true_f = tf.cast(y_true > threshold, tf.float32)
    y_pred_f = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def mean_iou(y_true, y_pred, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)
    scores = [iou_score(y_true, y_pred, float(t)) for t in thresholds]
    return tf.reduce_mean(tf.stack(scores))


def compute_iou(im1, im2, threshold=0.5, empty_score=1.0):
    im1 = np.asarray(im1 > threshold, dtype=bool)
    im2 = np.asarray(im2 > threshold, dtype=bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    union = np.logical_or(im1, im2).sum()
    if union == 0:
        return empty_score
    intersection = np.logical_and(im1, im2).sum()
    return intersection / float(union)


def compute_dice(im1, im2, threshold=0.5, empty_score=1.0):
    im1 = np.asarray(im1 > threshold, dtype=bool)
    im2 = np.asarray(im2 > threshold, dtype=bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    intersection = np.logical_and(im1, im2)
    return 2.0 * intersection.sum() / float(im_sum)
