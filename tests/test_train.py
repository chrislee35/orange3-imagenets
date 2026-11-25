from unittest import mock

import keras
import numpy as np
import pytest
from orangewidget.tests.base import WidgetTest

from orangecontrib.imagenets.util.image_loader import load_image, model_input_shape
from orangecontrib.imagenets.widgets import ow_image_train_and_score as ow_train
from orangecontrib.imagenets.widgets.ow_image_train_and_score import OWImageTrainAndScore
from orangecontrib.imagenets.widgets.ow_imagenet_classify import ClassifyWorker


def run_worker(worker):
    results = []
    worker.finished.connect(results.append)
    worker.run()
    return results


def small_model(shape=(28, 28, 1), classes=2):
    return keras.Sequential([
        keras.Input(shape), keras.layers.Rescaling(1 / 255), keras.layers.Flatten(),
        keras.layers.Dense(classes, activation="softmax"),
    ])


class TestImageLoader:
    def test_grayscale_and_color_shapes(self, image_table, tmp_path):
        path = str(tmp_path / image_table.metas[0, 0])
        gray = load_image(path, (28, 28, 1))
        color = load_image(path, (10, 20, 3))
        assert gray.shape == (28, 28, 1) and gray.dtype == np.float32
        assert color.shape == (10, 20, 3)
        assert gray.max() > 1  # raw 0..255, not scaled

    def test_unreadable_is_none(self, tmp_path):
        assert load_image(str(tmp_path / "missing.png"), (8, 8, 3)) is None

    def test_model_input_shape(self):
        assert model_input_shape(small_model((12, 16, 1))) == (12, 16, 1)
        assert model_input_shape(object()) == (224, 224, 3)


class TestTrain(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, image_table):
        self.image_table = image_table

    def setUp(self):
        self.w = self.create_widget(OWImageTrainAndScore)

    def test_trains_grayscale_with_validation(self):
        self.w.epochs, self.w.validation_percent = 2, 50
        self.w.model, self.w.data = small_model(), self.image_table
        self.w.train()
        model = self.get_output(self.w.Outputs.trained_model, widget=self.w)
        assert model.class_names == ["cat", "dog"]
        assert len(self.w.loss_values) == 2 and len(self.w.val_loss_values) == 2

    def test_class_count_mismatch_is_reported(self):
        self.w.model, self.w.data = small_model(classes=10), self.image_table
        self.w.train()
        assert self.get_output(self.w.Outputs.trained_model, widget=self.w) is None

    def test_classify_grayscale_model_uses_class_names(self):
        model = small_model()
        model.class_names = ["kitten", "puppy"]
        (predictions,) = run_worker(ClassifyWorker(model, self.image_table))
        assert len(predictions) == 4 and set(predictions) <= {"kitten", "puppy"}

    def test_validation_holds_out_every_class(self):
        # image tables list one class after another; the held-out share must still contain both classes
        seen = []
        model = small_model()
        real_fit = model.fit

        def spy(*a, **k):
            seen.append(k["validation_data"][1].argmax(axis=1))
            return real_fit(*a, **k)

        model.fit = spy
        self.w.epochs, self.w.validation_percent = 1, 50
        self.w.model, self.w.data = model, self.image_table
        with mock.patch.object(ow_train, "clone_model", lambda m: m):
            self.w.train()
        assert sorted(set(seen[0])) == [0, 1]
