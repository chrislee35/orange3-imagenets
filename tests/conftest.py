"""Shared fixtures; make ``orangecontrib.imagenets`` importable from a source checkout.

``orangecontrib`` is a namespace shared by every Orange add-on, and a copy installed next to Orange can win over a
checkout on ``sys.path``, so the checkout is appended to its search path.
"""

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Keep the developer's own Orange settings out of the tests.
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="imagenets-test-config-")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import orangecontrib  # noqa: E402
import pytest  # noqa: E402
from Orange.data import DiscreteVariable, Domain, StringVariable, Table  # noqa: E402

_PATH = os.path.join(ROOT, "orangecontrib")
if _PATH not in orangecontrib.__path__:
    orangecontrib.__path__.insert(0, _PATH)


@pytest.fixture
def image_table(tmp_path):
    """A 4-image Orange table in the layout produced by Image Analytics' Import Images widget."""
    rng = np.random.default_rng(0)
    names = []
    for i in range(4):
        label = "cat" if i % 2 == 0 else "dog"
        rel = os.path.join(label, f"img{i}.png")
        os.makedirs(tmp_path / label, exist_ok=True)
        cv2.imwrite(str(tmp_path / rel), rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))
        names.append(rel)
    image = StringVariable("image")
    image.attributes.update({"type": "image", "origin": str(tmp_path)})
    cls = DiscreteVariable("category", values=("cat", "dog"))
    domain = Domain([], cls, [image])
    return Table.from_numpy(
        domain,
        np.empty((4, 0)),
        np.array([0, 1, 0, 1], dtype=float),
        np.array(names, dtype=object).reshape(-1, 1),
    )
