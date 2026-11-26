"""Look inside a trained Keras image network: what its filters look like and what its layers respond to."""
import keras
import numpy as np

from orangecontrib.imagenets.util.image_loader import model_input_shape  # noqa: F401  (re-exported for widgets)

SHOWN = (keras.layers.Conv2D, keras.layers.MaxPooling2D, keras.layers.AveragePooling2D)


def feature_layers(model):
    """[(index, label)] of the top-level layers whose output is a stack of 2-D maps (convolutions, pooling, blocks)."""
    from orangecontrib.imagenets.util.layers import ResidualBlock

    out = []
    for i, layer in enumerate(model.layers):
        shape = getattr(layer, "output", None)
        shape = getattr(shape, "shape", None)
        if isinstance(layer, (*SHOWN, ResidualBlock)) and shape is not None and len(shape) == 4:
            out.append((i, f"{layer.name}  ({layer.__class__.__name__}, {shape[1]}x{shape[2]}x{shape[3]})"))
    return out


def _to_uint8(maps):
    """Stretch every map on its own to 0..255 (axis 0 indexes the maps)."""
    maps = np.asarray(maps, np.float32)
    flat = maps.reshape(len(maps), -1)
    lo, hi = flat.min(axis=1), flat.max(axis=1)
    scale = np.where(hi - lo > 1e-9, 255.0 / np.maximum(hi - lo, 1e-9), 0.0)
    out = (maps - lo.reshape(-1, 1, 1, 1)) * scale.reshape(-1, 1, 1, 1)
    return np.clip(out, 0, 255).astype(np.uint8)


def filter_images(layer):
    """The weights of a Conv2D layer as images, one per filter: uint8 (filters, kh, kw, 1 or 3).

    A filter on a color image has three kernels (red, green, blue) and is shown in color; on deeper layers, where a
    filter has one kernel per incoming map, the kernels are averaged to a single gray picture.
    """
    kernel = layer.get_weights()[0]  # (kh, kw, in, out)
    maps = np.transpose(kernel, (3, 0, 1, 2))  # (out, kh, kw, in)
    if maps.shape[-1] not in (1, 3):
        maps = maps.mean(axis=-1, keepdims=True)
    return _to_uint8(maps)


def feature_map_images(model, layer_index, image):
    """What layer ``layer_index`` outputs for one image (h, w, c, 0..255): uint8 (maps, h, w, 1), each stretched."""
    layer = model.layers[layer_index]
    probe = keras.Model(model.inputs, layer.output)
    out = probe.predict(np.asarray(image, np.float32)[None], verbose=0)[0]  # (h, w, maps)
    return _to_uint8(np.transpose(out, (2, 0, 1))[..., None])
