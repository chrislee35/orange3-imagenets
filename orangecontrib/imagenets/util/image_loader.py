"""Load images for Keras models, shared by Train ImageNet and Classify Images so both feed a model the same way.

Pixels are returned as float32 in 0..255; scaling to 0..1 is the job of the model's own Rescaling layer.
"""

import os

import cv2
import numpy as np

DEFAULT_SHAPE = (224, 224, 3)  # (height, width, channels) when a model does not state one


def model_input_shape(model, default=DEFAULT_SHAPE):
    """Return the (height, width, channels) a Keras model expects, or ``default`` if it does not say."""
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) == 4 and all(isinstance(d, int) for d in shape[1:]):
        return tuple(shape[1:])
    return default


def load_image(path, shape):
    """Read ``path`` as a float32 array of ``shape`` = (height, width, channels); None if it cannot be read."""
    height, width, channels = shape
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (width, height))
    if channels == 1:
        img = img[:, :, np.newaxis]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def load_table_images(data, shape, progress=None):
    """Load every image of an Orange image table.

    Returns ``(X, labels)``: ``X`` is float32 (n, height, width, channels) in 0..255; ``labels`` is a list of class
    names, or None when the table has no class variable. Unreadable images are skipped.
    """
    from orangecontrib.imagenets.util.image_table import image_table_variables

    origin, col = image_table_variables(data)
    class_var = data.domain.class_var
    images, labels = [], []
    for i, row in enumerate(data):
        img = load_image(os.path.join(origin, str(row.metas[col])), shape)
        if img is not None:
            images.append(img)
            if class_var is not None:
                labels.append(str(row.get_class()))
        if progress:
            progress((i + 1) / len(data))
    X = np.stack(images) if images else np.zeros((0, *shape), np.float32)
    return X, (labels if class_var is not None else None)


def to_uint8(images):
    """Map model output in -1..1 to uint8 pixels."""
    return np.clip((np.asarray(images) + 1.0) * 127.5, 0, 255).astype(np.uint8)


def to_signed(images):
    """Map pixels in 0..255 to the -1..1 range that generators and denoisers work in."""
    return np.asarray(images, np.float32) / 127.5 - 1.0


def image_grid(images, cols=None, pad=1):
    """Tile uint8 images (n, h, w, c) into one uint8 picture (rows*h, cols*w, c) for display."""
    images = np.asarray(images)
    n, h, w, c = images.shape
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    canvas = np.full((rows * (h + pad) - pad, cols * (w + pad) - pad, c), 40, np.uint8)
    for i, img in enumerate(images):
        r, k = divmod(i, cols)
        canvas[r * (h + pad): r * (h + pad) + h, k * (w + pad): k * (w + pad) + w] = img
    return canvas
