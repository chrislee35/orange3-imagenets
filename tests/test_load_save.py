import keras
import numpy as np
import pytest
from orangewidget.tests.base import WidgetTest

from orangecontrib.imagenets.widgets import ow_load_imagenet, ow_save_imagenet
from orangecontrib.imagenets.widgets.ow_load_imagenet import OWLoadKerasModel
from orangecontrib.imagenets.widgets.ow_save_imagenet import OWSaveImageNet


def tiny_model():
    return keras.Sequential([keras.Input((4,)), keras.layers.Dense(3, activation="softmax")])


class TestSave(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, monkeypatch, tmp_path):
        self.monkeypatch, self.tmp_path = monkeypatch, tmp_path

    def setUp(self):
        self.w = self.create_widget(OWSaveImageNet)

    def _choose(self, path):
        self.monkeypatch.setattr(ow_save_imagenet.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))

    def test_without_model_asks_nothing(self):
        def fail(*a, **k):
            raise AssertionError("dialog must not open without a model")

        self.monkeypatch.setattr(ow_save_imagenet.QFileDialog, "getSaveFileName", fail)
        self.w.save_as_h5()
        self.w.save_as_json()

    def test_save_h5_adds_extension(self):
        self._choose(self.tmp_path / "net")
        self.send_signal(self.w.Inputs.model, tiny_model(), widget=self.w)
        self.w.save_as_h5()
        assert (self.tmp_path / "net.h5").exists()
        assert self.w.last_dir == str(self.tmp_path)

    def test_save_json(self):
        self._choose(self.tmp_path / "net")
        self.send_signal(self.w.Inputs.model, tiny_model(), widget=self.w)
        self.w.save_as_json()
        assert '"Dense"' in (self.tmp_path / "net.json").read_text()

    def test_cancelled_dialog_writes_nothing(self):
        self._choose("")
        self.send_signal(self.w.Inputs.model, tiny_model(), widget=self.w)
        self.w.save_as_h5()
        assert not list(self.tmp_path.iterdir())


class TestLoad(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, monkeypatch, tmp_path):
        self.monkeypatch, self.tmp_path = monkeypatch, tmp_path

    def setUp(self):
        self.w = self.create_widget(OWLoadKerasModel)

    def test_h5_roundtrip(self):
        model = tiny_model()
        path = self.tmp_path / "net.h5"
        model.save(path)
        self.w.load_h5(str(path))
        loaded = self.get_output(self.w.Outputs.model, widget=self.w)
        x = np.ones((1, 4), dtype="float32")
        np.testing.assert_allclose(loaded.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-5)
        assert (self.w.load_type, self.w.load_file) == ("h5", str(path))

    def test_json_with_weights(self):
        model = tiny_model()
        (self.tmp_path / "net.json").write_text(model.to_json())
        model.save_weights(self.tmp_path / "net_weights.weights.h5")
        self.w.load_json(str(self.tmp_path / "net.json"))
        assert self.w.load_type == "json"

    def test_missing_file_sends_nothing(self):
        self.w.load_h5(str(self.tmp_path / "missing.h5"))
        assert self.get_output(self.w.Outputs.model, widget=self.w) is None
        assert ow_load_imagenet.load_model is not None
