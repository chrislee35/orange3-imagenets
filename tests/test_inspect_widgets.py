import keras
import numpy as np
import pytest
from Orange.data import DiscreteVariable, Domain, Table
from orangewidget.tests.base import WidgetTest

from orangecontrib.imagenets.util import pretrained
from orangecontrib.imagenets.util.confusion import confusion_matrix, true_and_predicted
from orangecontrib.imagenets.util.inspect import feature_layers, feature_map_images, filter_images
from orangecontrib.imagenets.widgets import ow_pretrained_imagenet
from orangecontrib.imagenets.widgets.ow_inspect_imagenet import OWInspectImageNet
from orangecontrib.imagenets.widgets.ow_misclassified import OWMisclassified
from orangecontrib.imagenets.widgets.ow_pretrained_imagenet import OWPretrainedImageNet


def cnn(channels=3):
    return keras.Sequential([
        keras.Input((16, 16, channels)), keras.layers.Rescaling(1 / 255),
        keras.layers.Conv2D(4, 3, activation="relu"), keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(6, 3), keras.layers.Flatten(), keras.layers.Dense(2, activation="softmax"),
    ])


def annotated(table, predictions):
    pred = DiscreteVariable("Prediction", values=tuple(sorted(set(predictions))))
    return table.add_column(pred, np.array([pred.values.index(p) for p in predictions], float), to_metas=True)


class TestInspectUtil:
    def test_feature_layers_lists_convs_and_pooling(self):
        names = [label.split()[0] for _, label in feature_layers(cnn())]
        assert len(names) == 3 and names[0].startswith("conv2d")

    def test_filter_images_color_and_deep(self):
        m = cnn()
        first = filter_images(m.layers[1])
        assert first.shape == (4, 3, 3, 3) and first.dtype == np.uint8
        assert filter_images(m.layers[3]).shape == (6, 3, 3, 1)  # 4 incoming maps averaged to one gray kernel

    def test_feature_maps(self):
        m = cnn()
        maps = feature_map_images(m, 1, np.random.rand(16, 16, 3) * 255)
        assert maps.shape == (4, 14, 14, 1)


class TestConfusion:
    def test_counts(self):
        labels, counts = confusion_matrix(["a", "a", "b", "b"], ["a", "b", "b", "b"])
        assert labels == ["a", "b"] and counts.tolist() == [[1, 1], [0, 2]]

    def test_needs_prediction_column(self, image_table):
        assert true_and_predicted(image_table) is None
        assert true_and_predicted(annotated(image_table, ["cat", "cat", "dog", "dog"])) == (
            ["cat", "dog", "cat", "dog"], ["cat", "cat", "dog", "dog"])


def test_prediction_stored_as_names(image_table):
    pred = DiscreteVariable("Prediction", values=("cat", "dog"))
    domain = Domain([], image_table.domain.class_var, [*image_table.domain.metas, pred])
    metas = np.column_stack([image_table.metas, np.array(["cat", "cat", "dog", "dog"], dtype=object)])
    table = Table.from_numpy(domain, np.empty((4, 0)), image_table.Y, metas)
    assert true_and_predicted(table)[1] == ["cat", "cat", "dog", "dog"]


class TestInspectWidget(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, image_table):
        self.image_table = image_table

    def test_draws_filters_and_maps(self):
        w = self.create_widget(OWInspectImageNet)
        self.send_signal(w.Inputs.model, cnn(), widget=w)
        assert w.layer_combo.count() == 3 and w.filter_label.pixmap() is not None
        self.send_signal(w.Inputs.data, self.image_table, widget=w)
        assert not w.Error.failed.is_shown()
        assert w.map_label.pixmap() is not None and w.input_label.pixmap() is not None

    def test_no_images_is_information_only(self):
        w = self.create_widget(OWInspectImageNet)
        self.send_signal(w.Inputs.model, cnn(), widget=w)
        assert w.Information.no_images.is_shown()


class TestMisclassifiedWidget(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, image_table):
        self.image_table = image_table

    def test_outputs_wrong_images_and_selection(self):
        w = self.create_widget(OWMisclassified)
        self.send_signal(w.Inputs.data, annotated(self.image_table, ["cat", "cat", "dog", "cat"]), widget=w)
        wrong = self.get_output(w.Outputs.misclassified, widget=w)
        assert len(wrong) == 3  # truth cat,dog,cat,dog against cat,cat,dog,cat
        w._cell_clicked(1, 0)  # true dog, predicted cat
        assert len(self.get_output(w.Outputs.selected, widget=w)) == 2
        assert "25.0% accuracy" in w.summary.text()

    def test_missing_prediction_is_an_error(self):
        w = self.create_widget(OWMisclassified)
        self.send_signal(w.Inputs.data, self.image_table, widget=w)
        assert w.Error.no_prediction.is_shown()


class TestPretrained(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, monkeypatch, image_table):
        monkeypatch.setattr(ow_pretrained_imagenet, "WEIGHTS", None)  # same network, no download
        self.image_table = image_table

    def test_head_and_freezing(self):
        w = self.create_widget(OWPretrainedImageNet)
        model = self.get_output(w.Outputs.model, widget=w)
        assert model.input_shape == (None, 96, 96, 3) and model.output_shape == (None, 10)
        trainable = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
        assert trainable == 1280 * 10 + 10  # only the new Dense layer
        w.freeze = False
        w.commit()
        assert sum(int(np.prod(v.shape)) for v in self.get_output(w.Outputs.model, widget=w).trainable_weights) > 100000

    def test_class_count_follows_data(self):
        w = self.create_widget(OWPretrainedImageNet)
        self.send_signal(w.Inputs.data, self.image_table, widget=w)
        assert w.n_classes == 2
        assert self.get_output(w.Outputs.model, widget=w).output_shape == (None, 2)

    def test_pretrained_bases_defined(self):
        assert all(size % 32 == 0 for _, size in pretrained.BASES.values())
