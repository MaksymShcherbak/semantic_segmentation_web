import numpy as np
import tensorflow as tf
import json
import pandas as pd
import tensorflow.keras as keras

def init_config(config_path):
    global CONFIG, CLASSES, CLASS_WEIGHTS
    with open(config_path, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
        CLASSES = CONFIG["CLASSES"]
        CLASS_WEIGHTS = tf.constant(
            [c["weight"] for c in CLASSES],
            dtype=tf.float32
        )
        return CONFIG

def unpack_palette(packed):
    r = (packed >> 16) & 255
    g = (packed >> 8)  & 255
    b = packed & 255
    return np.stack([r, g, b], axis=1)

def get_pallette(classes):
    palette = np.array([int(classes[i]["color"].lstrip("#"), 16) for i in range(len(classes))], dtype=np.uint32)
    return unpack_palette(palette)

# @keras.saving.register_keras_serializable()
class WeightedLoss(keras.losses.Loss):
    def __init__(self, name="weighted_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = CLASS_WEIGHTS
        self.sce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction="none"
        )

    def call(self, y_true, y_pred):
        y_true_s = tf.squeeze(y_true, axis=-1)
        y_true_s = tf.cast(y_true_s, tf.int32)

        weights = tf.gather(self.class_weights, y_true_s)
        per_pixel_loss = self.sce(y_true_s, y_pred)

        return tf.reduce_sum(per_pixel_loss * weights) / tf.reduce_sum(weights)

# @keras.saving.register_keras_serializable()
class WeightedAccuracy(keras.metrics.Metric):
    def __init__(self, name="weighted_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = CLASS_WEIGHTS

        self.total_weighted_matches = self.add_weight(
            name="tw_matches",
            initializer="zeros",
            shape=()
        )

        self.total_weights = self.add_weight(
            name="tw_total",
            initializer="zeros",
            shape=()
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_s = tf.squeeze(y_true, axis=-1)
        y_true_s = tf.cast(y_true_s, tf.int32)
        y_pred_cls = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        matches = tf.cast(tf.equal(y_true_s, y_pred_cls), tf.float32)
        weights = tf.gather(self.class_weights, y_true_s)

        self.total_weighted_matches.assign_add(tf.reduce_sum(matches * weights))
        self.total_weights.assign_add(tf.reduce_sum(weights))

    def result(self):
        return self.total_weighted_matches / (self.total_weights + 1e-7)

    def reset_state(self):
        self.total_weighted_matches.assign(0.0)
        self.total_weights.assign(0.0)

# @keras.saving.register_keras_serializable()
class NormalAccuracy(keras.metrics.Metric):
    def __init__(self, name="normal_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(
            name="correct",
            initializer="zeros",
            shape=()
        )

        self.total = self.add_weight(
            name="total",
            initializer="zeros",
            shape=()
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_s = tf.squeeze(y_true, axis=-1)
        y_true_s = tf.cast(y_true_s, tf.int32)
        y_pred_cls = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        matches = tf.cast(tf.equal(y_true_s, y_pred_cls), tf.float32)

        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(matches), tf.float32))

    def result(self):
        return self.correct / (self.total + 1e-7)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

# @keras.saving.register_keras_serializable()
class RandomSegmentation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        logits = tf.random.uniform(shape=(batch_size, CONFIG["TARGET_SIZE"][0],CONFIG["TARGET_SIZE"][1], len(CLASSES)))
        return tf.nn.softmax(logits, axis=-1)

# @keras.saving.register_keras_serializable()
class ResizeLike(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, target = inputs
        target_h, target_w = tf.shape(target)[1], tf.shape(target)[2]
        return tf.image.resize(x, (target_h, target_w), method='bilinear')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], input_shape[1][2], input_shape[0][3])