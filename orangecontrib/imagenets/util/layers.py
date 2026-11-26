"""Layers that Keras does not ship as one piece, for Build ImageNet's Sequential models."""
import keras


@keras.saving.register_keras_serializable(package="imagenets")
class ResidualBlock(keras.layers.Layer):
    """Two 3x3 convolutions with batch normalization, plus a skip connection that adds the block's input to its output.

    ``output = relu( BN(conv(relu(BN(conv(x))))) + x )``.  If ``filters`` differs from the input's channel count the
    skip goes through a 1x1 convolution so the two can be added.  A Sequential model cannot express this (it has no way
    to branch), so the block is one layer that branches inside.
    """

    def __init__(self, filters=64, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = keras.layers.Conv2D(filters, 3, padding="same")
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9)
        self.conv2 = keras.layers.Conv2D(filters, 3, padding="same")
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9)
        self.project = None
        self.add = keras.layers.Add()

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.project = keras.layers.Conv2D(self.filters, 1, padding="same")
            self.project.build(input_shape)
        self.conv1.build(input_shape)
        shape = self.conv1.compute_output_shape(input_shape)
        self.bn1.build(shape)
        self.conv2.build(shape)
        self.bn2.build(shape)

    def call(self, x, training=None):
        y = keras.activations.relu(self.bn1(self.conv1(x), training=training))
        y = self.bn2(self.conv2(y), training=training)
        skip = x if self.project is None else self.project(x)
        return keras.activations.relu(self.add([y, skip]))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.filters)

    def get_config(self):
        return {**super().get_config(), "filters": self.filters}
