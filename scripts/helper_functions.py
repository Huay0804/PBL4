import numpy as np
import tensorflow as tf

BOUNDARY_LOSS_WEIGHT = 4.0
BOUNDARY_LOSS_DILATION = 1


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
    y_pred = tf.cast(y_pred, tf.float32)
    num_classes = tf.shape(y_pred)[-1]

    if y_true.shape.rank is not None and y_true.shape.rank == y_pred.shape.rank:
        if y_true.shape[-1] == 1:
            labels = tf.squeeze(y_true, axis=-1)
        else:
            labels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    else:
        labels = y_true

    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, num_classes])

    pred_sum = tf.reduce_sum(y_pred_flat, axis=0)
    true_sum = tf.cast(
        tf.math.bincount(labels, minlength=num_classes, maxlength=num_classes),
        tf.float32,
    )
    gather_indices = tf.stack([tf.range(tf.shape(labels)[0], dtype=tf.int32), labels], axis=1)
    true_class_probs = tf.gather_nd(y_pred_flat, gather_indices)
    intersection = tf.math.unsorted_segment_sum(true_class_probs, labels, num_classes)

    denom = true_sum + pred_sum
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


def _labels_from_mask(y_true):
    if y_true.shape.rank is not None and y_true.shape.rank == 4:
        if y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        else:
            y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    return tf.cast(y_true, tf.int32)


def boundary_weight_map(y_true, dilation=BOUNDARY_LOSS_DILATION):
    labels = _labels_from_mask(y_true)[..., tf.newaxis]

    vertical = tf.cast(tf.not_equal(labels[:, 1:, :, :], labels[:, :-1, :, :]), tf.float32)
    horizontal = tf.cast(tf.not_equal(labels[:, :, 1:, :], labels[:, :, :-1, :]), tf.float32)

    boundary = tf.zeros_like(tf.cast(labels, tf.float32))
    boundary += tf.pad(vertical, [[0, 0], [1, 0], [0, 0], [0, 0]])
    boundary += tf.pad(vertical, [[0, 0], [0, 1], [0, 0], [0, 0]])
    boundary += tf.pad(horizontal, [[0, 0], [0, 0], [1, 0], [0, 0]])
    boundary += tf.pad(horizontal, [[0, 0], [0, 0], [0, 1], [0, 0]])
    boundary = tf.clip_by_value(boundary, 0.0, 1.0)

    if dilation > 0:
        kernel = 2 * dilation + 1
        boundary = tf.nn.max_pool2d(
            boundary,
            ksize=(kernel, kernel),
            strides=(1, 1),
            padding="SAME",
        )
    return boundary


def ce_dice_boundary_loss(y_true, y_pred):
    weights = 1.0 + BOUNDARY_LOSS_WEIGHT * tf.squeeze(boundary_weight_map(y_true), axis=-1)

    if y_pred.shape.rank is not None and y_pred.shape[-1] == 1:
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        ce = tf.reduce_mean(ce * weights)
        return ce + dice_coef_loss(y_true, y_pred)

    labels = _labels_from_mask(y_true)
    ce = tf.keras.losses.sparse_categorical_crossentropy(labels, y_pred)
    ce = tf.reduce_mean(ce * weights)
    return ce + multiclass_dice_loss(labels, y_pred)


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
