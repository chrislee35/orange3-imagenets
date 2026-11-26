import keras
import numpy as np

from orangecontrib.imagenets.util.layers import ResidualBlock
from orangecontrib.imagenets.widgets.ow_imagenet_builder import OWImageNetBuilder
from orangewidget.tests.base import WidgetTest


def test_block_keeps_shape_and_adds_skip():
    x = np.random.rand(2, 8, 8, 4).astype("float32")
    assert ResidualBlock(4)(x).shape == (2, 8, 8, 4)
    assert ResidualBlock(6)(x).shape == (2, 8, 8, 6)  # 1x1 projection on the skip


def test_save_and_load_roundtrip(tmp_path):
    m = keras.Sequential([keras.Input((8, 8, 1)), keras.layers.Conv2D(4, 3, padding="same"), ResidualBlock(4),
                          keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(3, activation="softmax")])
    x = np.random.rand(2, 8, 8, 1).astype("float32")
    path = str(tmp_path / "m.keras")
    m.save(path)
    m2 = keras.models.load_model(path)
    assert np.allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), atol=1e-5)


class TestBuilderResidual(WidgetTest):
    def test_miniresnet_builds(self):
        w = self.create_widget(OWImageNetBuilder)
        w.input_height = w.input_width = 28
        w.channels_index = 0
        w.load_prebuilt_model("MiniResNet")
        model = self.get_output(w.Outputs.model, widget=w)
        assert model is not None and model.output_shape == (None, 10)
        assert any(isinstance(layer, ResidualBlock) for layer in model.layers)
        print(w.size_label.text())
