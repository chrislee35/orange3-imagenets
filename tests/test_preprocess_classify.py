import os

import cv2
import keras
import pytest
from orangewidget.tests.base import WidgetTest

from orangecontrib.imagenets.widgets.ow_image_preprocessor import OWImagePreprocessor, PreprocessWorker
from orangecontrib.imagenets.widgets.ow_imagenet_classify import ClassifyWorker, OWImageNetClassify


def run_worker(worker):
    """Run a QThread worker's body synchronously and return what it emitted."""
    results = []
    worker.finished.connect(results.append)
    worker.run()
    return results


class TestPreprocessWorker:
    def test_resizes_and_keeps_folder_layout(self, image_table, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (result,) = run_worker(PreprocessWorker(image_table, str(out), False, True, 16, 8, False))
        for rel in image_table.metas[:, 0]:
            img = cv2.imread(os.path.join(out, rel))
            assert img.shape[:2] == (8, 16)
        assert result.domain.metas[0].attributes["origin"] == str(out)

    def test_grayscale_and_normalize(self, image_table, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        run_worker(PreprocessWorker(image_table, str(out), True, False, 0, 0, True))
        img = cv2.imread(os.path.join(out, image_table.metas[0, 0]), cv2.IMREAD_UNCHANGED)
        assert img.ndim == 2 and img.min() == 0 and img.max() == 255


class TestPreprocessWidget(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, image_table, tmp_path):
        self.image_table, self.tmp_path = image_table, tmp_path

    def setUp(self):
        self.w = self.create_widget(OWImagePreprocessor)

    def test_no_data(self):
        self.send_signal(self.w.Inputs.data, None, widget=self.w)
        assert self.w.info_label.text() == "No data"

    def test_preview_scaled_to_settings(self):
        self.w.resize_width, self.w.resize_height = 20, 10
        self.send_signal(self.w.Inputs.data, self.image_table, widget=self.w)
        pixmap = self.w.aug_image_label.pixmap()
        assert (pixmap.width(), pixmap.height()) == (20, 10)

    def test_preprocess_image_grayscale(self):
        self.w.do_grayscale = True
        self.w.do_resize = False
        pixmap = self.w.preprocess_image(os.path.join(self.tmp_path, self.image_table.metas[0, 0]))
        assert (pixmap.width(), pixmap.height()) == (60, 40)

    def test_handle_preprocessed_sends_output(self):
        self.w.handle_preprocessed(self.image_table)
        assert self.get_output(self.w.Outputs.preprocessed_data, widget=self.w) is self.image_table
        assert self.w.info_label.text() == "Preprocessing complete."


class TestClassify(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, image_table, tmp_path):
        self.image_table, self.tmp_path = image_table, tmp_path

    @staticmethod
    def model():
        return keras.Sequential([keras.Input((224, 224, 3)), keras.layers.Flatten(), keras.layers.Dense(2, activation="softmax")])

    def test_worker_predicts_one_class_per_image(self):
        (predictions,) = run_worker(ClassifyWorker(self.model(), self.image_table))
        assert len(predictions) == 4
        assert set(predictions) <= {"cat", "dog"}

    def test_widget_waits_for_both_inputs(self):
        w = self.create_widget(OWImageNetClassify)
        self.send_signal(w.Inputs.data, self.image_table, widget=w)
        assert w.info_label.text() == "Waiting for input..."

    def test_handle_results_adds_prediction_meta(self):
        w = self.create_widget(OWImageNetClassify)
        w.data = self.image_table
        w.handle_results(["cat", "dog", "dog", "cat"])
        out = self.get_output(w.Outputs.annotated_data, widget=w)
        assert "Prediction" in [m.name for m in out.domain.metas]
        truth = [self.image_table.domain.class_var.values[int(v)] for v in self.image_table.Y]
        right = sum(t == p for t, p in zip(truth, ["cat", "dog", "dog", "cat"]))
        assert w.accuracy == right / 4
        assert w.info_label.text() == f"Classified 4 images. Accuracy: {right / 4:.1%}"
