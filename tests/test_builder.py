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
        self.w.load_prebuilt_model("SimpleCNN")  # Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
        self.w._move_layer(1, 1)
        assert self.w.model_layers[1]["type"] == "MaxPooling2D"
        self.w._move_layer(2, -1)  # swaps back
        assert self.w.model_layers[1]["type"] == "Conv2D"
        self.w._delete_layer(1)
        assert len(self.w.model_layers) == 4

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


    def test_presets_start_with_rescaling_and_are_not_shared(self):
        self.w.load_prebuilt_model("SimpleCNN")
        assert self.w.model_layers[0]["type"] == "Rescaling"
        self.w._update_param(self.w.model_layers[-1], "units", 26)
        assert PREBUILT_MODELS["SimpleCNN"][-1]["units"] == 10

    def test_input_shape_follows_settings(self):
        self.w.load_prebuilt_model("SimpleCNN")
        assert self.get_output(self.w.Outputs.model, widget=self.w).input_shape == (None, 224, 224, 3)
        self.w.input_height = self.w.input_width = 28
        self.w.channels_index = 0
        self.w._update_model_config()
        assert self.get_output(self.w.Outputs.model, widget=self.w).input_shape == (None, 28, 28, 1)

    def test_size_label_shows_weight_count(self):
        self.w.input_height = self.w.input_width = 28
        self.w.channels_index = 0
        self.w.load_prebuilt_model("SimpleCNN")
        assert self.w.size_label.text().startswith("Weights: 54,410")
        self.w.load_prebuilt_model("LeNet")
        assert "Weights:" in self.w.size_label.text()
