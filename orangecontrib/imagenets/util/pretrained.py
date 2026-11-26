"""A small ImageNet-pretrained network (MobileNetV2) with a fresh classification head, for transfer learning."""
import keras

# label -> (alpha, input size). Both weight files are small (2 to 4 MB) and are downloaded by Keras on first use.
BASES = {
    "MobileNetV2 small (96 x 96)": (0.35, 96),
    "MobileNetV2 medium (128 x 128)": (0.5, 128),
}


def build_pretrained(base_label, n_classes, freeze=True, weights="imagenet", dropout=0.2):
    """Sequential model: pixels 0..255 -> pretrained convolutional base -> pooling -> dropout -> Dense(n_classes).

    ``freeze`` keeps the base's weights fixed so only the new head learns (fast, needs little data); without it the whole
    network is fine-tuned. ``weights=None`` builds the same architecture with random weights (used by the tests).
    """
    alpha, size = BASES[base_label]
    base = keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=alpha, include_top=False, weights=weights)
    base.trainable = not freeze
    return keras.Sequential([
        keras.Input((size, size, 3)),
        keras.layers.Rescaling(1 / 127.5, offset=-1),  # the range MobileNetV2 was trained with
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(n_classes, activation="softmax"),
    ], name="pretrained")
