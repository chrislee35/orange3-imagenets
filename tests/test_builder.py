import json

import keras
from orangewidget.tests.base import WidgetTest

from orangecontrib.imagenets.widgets.ow_imagenet_builder import PREBUILT_MODELS, OWImageNetBuilder


class TestImageNetBuilder(WidgetTest):
    def setUp(self):
        self.w = self.create_widget(OWImageNetBuilder)

    def test_prebuilt_models_are_valid(self):
        for name, layers in PREBUILT_MODELS.items():
            self.w.load_prebuilt_model(name)
            assert self.w.model_layers == layers
            model = self.get_output(self.w.Outputs.model, widget=self.w)
            assert isinstance(model, keras.Sequential)
            assert len(model.layers) == len(layers)

    def test_add_layer_inserts_before_output(self):
        self.w.load_prebuilt_model("SimpleCNN")
        self.w.add_layer("Dropout")
        types = [layer["type"] for layer in self.w.model_layers]
        assert types[-2:] == ["Dropout", "Dense"]

    def test_clear_layers_resets_to_rescaling_and_dense(self):
        self.w.load_prebuilt_model("LeNet")
        self.w.clear_layers()
        assert [layer["type"] for layer in self.w.model_layers] == ["Rescaling", "Dense"]
        assert self.w.model_layers[-1]["activation"] == "softmax"

    def test_move_and_delete(self):
        self.w.load_prebuilt_model("SimpleCNN")
        self.w._move_layer(0, 1)
        assert self.w.model_layers[0]["type"] == "MaxPooling2D"
        self.w._move_layer(0, -1)  # out of range: no-op
        assert self.w.model_layers[0]["type"] == "MaxPooling2D"
        self.w._delete_layer(0)
        assert len(self.w.model_layers) == 3

    def test_update_param_rebuilds_model(self):
        self.w.load_prebuilt_model("SimpleCNN")
        self.w._update_param(self.w.model_layers[-1], "units", 3)
        model = self.get_output(self.w.Outputs.model, widget=self.w)
        assert model.layers[-1].units == 3

    def test_config_is_saved_as_json(self):
        self.w.load_prebuilt_model("SimpleCNN")
        assert json.loads(self.w.model_config) == PREBUILT_MODELS["SimpleCNN"]

    def test_invalid_layer_sends_none(self):
        self.w.model_layers = [{"type": "NoSuchLayer"}]
        self.w._update_model_config()
        assert self.get_output(self.w.Outputs.model, widget=self.w) is None

    def test_default_config(self):
        assert self.w.default_config("Dropout") == {"type": "Dropout", "rate": 0.5}
        assert self.w.default_config("Unknown") == {"type": "Unknown"}
