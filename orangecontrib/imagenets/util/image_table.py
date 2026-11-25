from Orange.data import Table


def image_table_variables(data: Table) -> [str, int]:
    domain = data.domain
    image_col = None
    origin = None
    for var in domain.metas:
        if var.attributes.get("type") == "image":
            image_col = var
            origin = var.attributes.get("origin")
            break
    if image_col is None:
        raise Exception("No variable with type \"image\"")
    image_col_index = domain.metas.index(image_col)
    return origin, image_col_index


def images_to_table(images, folder, labels=None, prefix="gen"):
    """Write uint8 images (n, h, w, c) as PNGs into ``folder`` and return an Orange image table that points at them.

    The table has the same layout as Image Analytics' *Import Images* produces (a string meta variable marked
    ``type=image`` with an ``origin``), so Image Viewer, Train ImageNet and Classify Images can use it. ``labels``
    (optional) become the class variable.
    """
    import os

    import cv2
    import numpy as np
    from Orange.data import DiscreteVariable, Domain, StringVariable

    os.makedirs(folder, exist_ok=True)
    names = []
    for i, img in enumerate(images):
        name = f"{prefix}_{i:04d}.png"
        cv2.imwrite(os.path.join(folder, name), img[:, :, 0] if img.shape[2] == 1 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        names.append(name)
    image = StringVariable("image")
    image.attributes.update({"type": "image", "origin": folder})
    cls, y = None, None
    if labels is not None:
        values = sorted(set(labels))
        cls = DiscreteVariable("category", values=values)
        y = np.array([values.index(v) for v in labels], dtype=float)
    domain = Domain([], cls, [image])
    return Table.from_numpy(domain, np.empty((len(names), 0)), y, np.array(names, dtype=object).reshape(-1, 1))
