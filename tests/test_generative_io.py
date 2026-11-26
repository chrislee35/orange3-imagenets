import numpy as np
import pytest
from orangewidget.tests.base import WidgetTest

from orangecontrib.imagenets.util.diffusion import train_diffusion
from orangecontrib.imagenets.util.gan import train_gan
from orangecontrib.imagenets.util.generative_io import is_generative, load_generative, save_generative
from orangecontrib.imagenets.widgets import ow_load_imagenet, ow_save_imagenet
from orangecontrib.imagenets.widgets.ow_load_imagenet import OWLoadKerasModel
from orangecontrib.imagenets.widgets.ow_save_imagenet import OWSaveImageNet

IMAGES = np.random.default_rng(0).integers(0, 256, (12, 8, 8, 1)).astype(np.float32)


@pytest.fixture(scope="module")
def gan():
    return train_gan(IMAGES, epochs=1, batch_size=4, latent_dim=8, base_filters=4)


@pytest.fixture(scope="module")
def diffusion():
    return train_diffusion(IMAGES, ["a", "b"] * 6, epochs=1, batch_size=4, steps=5, base_filters=4)


class TestRoundtrip:
    def test_gan_same_images_after_reload(self, gan, tmp_path):
        save_generative(gan, tmp_path / "g.imagenet")
        again = load_generative(tmp_path / "g.imagenet")
        assert again.kind == "gan" and again.shape == gan.shape and again.latent_dim == 8
        assert (again.sample(4, seed=2) == gan.sample(4, seed=2)).all()

    def test_diffusion_same_images_after_reload(self, diffusion, tmp_path):
        save_generative(diffusion, tmp_path / "d.imagenet")
        again = load_generative(tmp_path / "d.imagenet")
        assert again.kind == "diffusion" and again.class_names == ["a", "b"] and again.steps == 5
        assert (again.sample(3, labels=1, seed=4) == diffusion.sample(3, labels=1, seed=4)).all()

    def test_is_generative(self, gan):
        assert is_generative(gan) and not is_generative(object())


class TestWidgets(WidgetTest):
    @pytest.fixture(autouse=True)
    def _inject(self, monkeypatch, tmp_path, gan):
        self.mp, self.tmp_path, self.gan = monkeypatch, tmp_path, gan

    def test_save_then_load_widgets(self):
        path = self.tmp_path / "net"
        self.mp.setattr(ow_save_imagenet.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
        save = self.create_widget(OWSaveImageNet)
        self.send_signal(save.Inputs.model, self.gan, widget=save)
        save.save_generative_model()
        assert (self.tmp_path / "net.imagenet").exists()
        load = self.create_widget(OWLoadKerasModel)
        self.mp.setattr(ow_load_imagenet.QFileDialog, "getOpenFileName", lambda *a, **k: (str(self.tmp_path / "net.imagenet"), ""))
        load.load_generative_dialog()
        out = self.get_output(load.Outputs.model, widget=load)
        assert out.kind == "gan" and load.load_type == "imagenet"

    def test_h5_save_refuses_generative_model(self):
        def fail(*a, **k):
            raise AssertionError("no dialog expected")

        self.mp.setattr(ow_save_imagenet.QFileDialog, "getSaveFileName", fail)
        save = self.create_widget(OWSaveImageNet)
        self.send_signal(save.Inputs.model, self.gan, widget=save)
        save.save_as_h5()
        save.save_as_json()
