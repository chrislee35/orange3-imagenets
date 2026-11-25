"""Pieces shared by the generative widgets (Train GAN, Train Diffusion, Generate Images)."""

import numpy as np
import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QImage, QPixmap
from AnyQt.QtWidgets import QLabel
from Orange.widgets import gui

from orangecontrib.imagenets.util.image_loader import image_grid


def pixmap_from_array(img, scale=1):
    """uint8 (h, w, 1|3) array -> QPixmap, enlarged ``scale`` times without smoothing."""
    img = np.ascontiguousarray(img)
    h, w, c = img.shape
    fmt = QImage.Format_Grayscale8 if c == 1 else QImage.Format_RGB888
    qimg = QImage(img.data, w, h, w * c, fmt).copy()
    return QPixmap.fromImage(qimg).scaled(w * scale, h * scale, Qt.KeepAspectRatio, Qt.FastTransformation)


def show_images(label: QLabel, images, cols=None, max_width=480):
    """Tile ``images`` (uint8 n x h x w x c) into ``label``, enlarging small images to fill ``max_width``."""
    grid = image_grid(images, cols)
    scale = max(1, max_width // grid.shape[1])
    label.setPixmap(pixmap_from_array(grid, scale))


def shape_controls(widget, box_title="Training images"):
    """Height, Width and Channels controls bound to widget settings ``img_height``, ``img_width``, ``channels_index``."""
    box = gui.widgetBox(widget.controlArea, box_title)
    gui.spin(box, widget, "img_height", 4, 512, step=4, label="Height:")
    gui.spin(box, widget, "img_width", 4, 512, step=4, label="Width:")
    gui.comboBox(box, widget, "channels_index", label="Channels:", items=["1 (grayscale)", "3 (color)"])
    return box


def channels(widget):
    return 1 if widget.channels_index == 0 else 3


def make_loss_plot(title="Training Progress"):
    plot = pg.PlotWidget(title=title)
    plot.setLabel("bottom", "Epoch")
    plot.showGrid(x=True, y=True)
    plot.addLegend()
    return plot
