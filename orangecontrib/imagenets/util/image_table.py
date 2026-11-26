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


def retarget_images(table, origin, rows=None, image_names=None):
    """Return a copy of an image table whose image column points at ``origin``.

    The input table is left alone: variables (and their ``attributes``) are shared by every widget that holds the
    table, so changing ``origin`` in place would silently break the original table downstream.
    ``rows`` and ``image_names`` (optional) replace the table's rows / image-column values.
    """
    import numpy as np
    from Orange.data import Domain

    _, col = image_table_variables(table)
    metas = list(table.domain.metas)
    image = metas[col].copy()
    image.attributes = dict(metas[col].attributes, origin=origin)
    metas[col] = image
    domain = Domain(table.domain.attributes, table.domain.class_vars, metas)
    out = table.transform(domain) if rows is None else table[rows].transform(domain)
    if image_names is not None:
        m = np.array(out.metas, dtype=object)
        m[:, col] = image_names
        out = out.from_numpy(domain, out.X, out.Y, m)
    return out


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
