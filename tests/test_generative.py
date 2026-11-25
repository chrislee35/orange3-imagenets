import os

import numpy as np
import pytest
from orangewidget.tests.base import WidgetTest

from orangecontrib.imagenets.util.diffusion import DiffusionModel, make_schedule, time_embedding, train_diffusion
from orangecontrib.imagenets.util.gan import GANModel, train_gan
from orangecontrib.imagenets.util.image_loader import image_grid, to_signed, to_uint8
from orangecontrib.imagenets.util.image_table import images_to_table
from orangecontrib.imagenets.widgets.ow_generate_images import OWGenerateImages
from orangecontrib.imagenets.widgets.ow_train_diffusion import OWTrainDiffusion
from orangecontrib.imagenets.widgets.ow_train_gan import OWTrainGAN


@pytest.fixture(scope="module")
def tiny_images():
    return np.random.default_rng(0).integers(0, 256, (12, 8, 8, 1)).astype(np.float32)


class TestHelpers:
    def test_range_conversions_roundtrip(self):
        px = np.array([0, 128, 255], np.float32)
        assert to_uint8(to_signed(px)).tolist() == [0, 128, 255]

    def test_image_grid_shape(self):
        grid = image_grid(np.zeros((6, 4, 5, 1), np.uint8), cols=3, pad=1)
        assert grid.shape == (2 * 4 + 1, 3 * 5 + 2, 1)

    def test_schedule_ends_nearly_pure_noise_for_any_step_count(self):
        for steps in (50, 200, 1000):
            _, _, ab = make_schedule(steps)
            assert ab[0] > 0.95 and ab[-1] < 0.01

    def test_time_embedding_shape_and_distinct(self):
        e = time_embedding([0, 1, 100])
        assert e.shape == (3, 64) and not np.allclose(e[0], e[1])

    def test_images_to_table_matches_import_images_layout(self, tmp_path):
        imgs = np.zeros((3, 8, 8, 1), np.uint8)
        table = images_to_table(imgs, str(tmp_path), labels=["a", "b", "a"])
        var = table.domain.metas[0]
        assert var.attributes == {"type": "image", "origin": str(tmp_path)}
        assert [os.path.exists(tmp_path / str(m)) for m in table.metas[:, 0]] == [True] * 3
        assert list(table.domain.class_var.values) == ["a", "b"]


class TestGAN:
    def test_trains_and_samples(self, tiny_images):
        seen = []
        model = train_gan(tiny_images, epochs=2, batch_size=4, latent_dim=8, base_filters=4,
                          on_epoch=lambda *a: seen.append(a))
        assert isinstance(model, GANModel) and len(seen) == 2
        assert seen[0][4].shape == (16, 8, 8, 1)
        out = model.sample(5, seed=3)
        assert out.shape == (5, 8, 8, 1) and out.dtype == np.uint8
        assert (model.sample(2, seed=3) == model.sample(2, seed=3)).all()

    def test_stop(self, tiny_images):
        calls = []
        train_gan(tiny_images, epochs=5, batch_size=4, latent_dim=8, base_filters=4, on_epoch=lambda *a: calls.append(1),
                  should_stop=lambda: True)
        assert len(calls) == 1

    def test_rejects_size_not_divisible_by_4(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            train_gan(np.zeros((4, 10, 10, 1), np.float32), epochs=1)

    def test_discriminator_actually_learns(self, tiny_images):
        # regression: Keras 3 silently froze the discriminator in the classic compiled "combined model" setup
        losses = []
        train_gan(tiny_images, epochs=6, batch_size=4, latent_dim=8, base_filters=4, on_epoch=lambda e, d, *a: losses.append(d))
        assert losses[-1] != pytest.approx(losses[0])


class TestDiffusion:
    def test_trains_conditioned_and_samples(self, tiny_images):
        labels = ["a", "b"] * 6
        seen = []
        model = train_diffusion(tiny_images, labels, epochs=2, batch_size=4, steps=10, base_filters=4,
                                on_epoch=lambda e, loss, m: seen.append(loss))
        assert isinstance(model, DiffusionModel) and len(seen) == 2 and model.class_names == ["a", "b"]
        out, snaps = model.sample(3, labels=[0, 1, 0], guidance=1.0, seed=1, keep_steps=4)
        assert out.shape == (3, 8, 8, 1) and len(snaps) == 4
        assert (snaps[-1] == out).all()
        assert (model.sample(2, labels=1, seed=5) == model.sample(2, labels=1, seed=5)).all()

    def test_unconditioned(self, tiny_images):
        model = train_diffusion(tiny_images, None, epochs=1, batch_size=4, steps=5, base_filters=4)
        assert model.class_names is None and model.sample(2, seed=0).shape == (2, 8, 8, 1)


class TestWidgets(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, image_table):
        self.image_table = image_table  # 4 color images 40 wide x 60... stored 60x40; loader resizes

    def _set_small(self, w):
        w.img_height = w.img_width = 8
        w.channels_index = 0
        w.epochs, w.batch_size, w.base_filters = 1, 4, 4

    def test_train_gan_widget_outputs_generator(self):
        w = self.create_widget(OWTrainGAN)
        self._set_small(w)
        w.latent_dim = 8
        self.send_signal(w.Inputs.data, self.image_table, widget=w)
        w.train()
        model = self.get_output(w.Outputs.generator, widget=w)
        assert model.kind == "gan" and model.shape == (8, 8, 1)
        assert len(w.d_losses) == 1

    def test_train_gan_widget_reports_bad_size(self):
        w = self.create_widget(OWTrainGAN)
        self._set_small(w)
        w.img_height = 10
        self.send_signal(w.Inputs.data, self.image_table, widget=w)
        w.train()
        assert w.Error.bad_shape.is_shown()
        assert self.get_output(w.Outputs.generator, widget=w) is None

    def test_train_diffusion_widget_and_generate(self):
        w = self.create_widget(OWTrainDiffusion)
        self._set_small(w)
        w.noise_steps, w.preview_every = 5, 1
        self.send_signal(w.Inputs.data, self.image_table, widget=w)
        w.train()
        model = self.get_output(w.Outputs.model, widget=w)
        assert model.class_names == ["cat", "dog"]
        g = self.create_widget(OWGenerateImages)
        g.count, g.show_steps = 3, True
        self.send_signal(g.Inputs.model, model, widget=g)
        out = self.get_output(g.Outputs.images, widget=g)
        assert len(out) == 3
        g.class_index = 2  # "dog"
        g.generate()
        out = self.get_output(g.Outputs.images, widget=g)
        assert [out.domain.class_var.str_val(v) for v in out.Y] == ["dog"] * 3

    def test_generate_with_gan_disables_diffusion_controls(self):
        tiny = np.random.default_rng(0).integers(0, 256, (12, 8, 8, 1)).astype(np.float32)
        model = train_gan(tiny, epochs=1, batch_size=4, latent_dim=8, base_filters=4)
        g = self.create_widget(OWGenerateImages)
        g.count = 4
        self.send_signal(g.Inputs.model, model, widget=g)
        assert not g.class_combo.isEnabled() and not g.steps_check.isEnabled()
        assert g.class_combo.itemText(0) == "(GANs have no classes)"
        assert len(self.get_output(g.Outputs.images, widget=g)) == 4
