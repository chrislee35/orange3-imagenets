import os

import pytest
from orangewidget.tests.base import WidgetTest

from orangecontrib.imagenets.util.image_table import retarget_images
from orangecontrib.imagenets.widgets.ow_image_augmenter import OWImageAugmenter


def origin_of(table):
    return table.domain.metas[0].attributes["origin"]


def test_retarget_leaves_input_alone(image_table, tmp_path):
    before = origin_of(image_table)
    out = retarget_images(image_table, str(tmp_path))
    assert origin_of(image_table) == before
    assert origin_of(out) == str(tmp_path)
    assert len(out) == len(image_table)


def test_retarget_rows_and_names(image_table, tmp_path):
    out = retarget_images(image_table, str(tmp_path), rows=[0, 0, 1], image_names=["a.png", "b.png", "c.png"])
    assert list(out.metas[:, 0]) == ["a.png", "b.png", "c.png"]
    assert out.Y[0] == out.Y[1] == image_table.Y[0]


class TestAugmenter(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, image_table, tmp_path):
        self.image_table, self.tmp_path = image_table, tmp_path

    def setUp(self):
        self.w = self.create_widget(OWImageAugmenter)

    def run_it(self, folder, originals):
        self.w.save_folder, self.w.augment_count, self.w.include_originals = str(folder), 2, originals
        self.send_signal(self.w.Inputs.images, self.image_table, widget=self.w)
        self.w.generate_augmentations()
        return self.get_output(self.w.Outputs.augmented_images, widget=self.w)

    def test_originals_plus_augmented_and_input_untouched(self):
        before = origin_of(self.image_table)
        out = self.run_it(self.tmp_path / "aug", True)
        assert len(out) == len(self.image_table) * 3
        assert origin_of(self.image_table) == before
        assert origin_of(out) == str(self.tmp_path / "aug")
        assert all(os.path.exists(os.path.join(origin_of(out), n)) for n in out.metas[:, 0])

    def test_augmented_only(self):
        out = self.run_it(self.tmp_path / "aug", False)
        assert len(out) == len(self.image_table) * 2

    def test_flip_off_by_default(self):
        assert self.w.flip is False

    def test_no_folder_sends_nothing(self):
        out = self.run_it("", True)
        assert out is None
