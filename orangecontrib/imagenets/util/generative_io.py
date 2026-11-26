"""Save and load GAN and diffusion models as a single ``.imagenet`` file.

The file is a zip holding ``meta.json`` (what kind of model, its image shape and settings) and ``net.keras`` (the Keras
network: the generator, or the denoiser).
"""

import json
import os
import tempfile
import zipfile

import keras

from orangecontrib.imagenets.util import layers as _custom_layers  # noqa: F401  (registers ResidualBlock)

from orangecontrib.imagenets.util.diffusion import DiffusionModel
from orangecontrib.imagenets.util.gan import GANModel

EXTENSION = ".imagenet"


def is_generative(model):
    return getattr(model, "kind", None) in ("gan", "diffusion")


def save_generative(model, path):
    if model.kind == "gan":
        net, meta = model.generator, {"kind": "gan", "latent_dim": model.latent_dim, "shape": list(model.shape)}
    elif model.kind == "diffusion":
        net = model.net
        meta = {"kind": "diffusion", "shape": list(model.shape), "steps": model.steps, "class_names": model.class_names}
    else:
        raise ValueError(f"Cannot save a model of kind {model.kind!r}.")
    with tempfile.TemporaryDirectory() as tmp:
        net_path = os.path.join(tmp, "net.keras")
        net.save(net_path)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("meta.json", json.dumps(meta))
            z.write(net_path, "net.keras")


def load_generative(path):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json"))
        with tempfile.TemporaryDirectory() as tmp:
            z.extract("net.keras", tmp)
            net = keras.models.load_model(os.path.join(tmp, "net.keras"), compile=False)
    shape = tuple(meta["shape"])
    if meta["kind"] == "gan":
        return GANModel(net, meta["latent_dim"], shape)
    if meta["kind"] == "diffusion":
        return DiffusionModel(net, shape, meta["steps"], meta["class_names"])
    raise ValueError(f"Unknown model kind {meta['kind']!r} in {path}.")
