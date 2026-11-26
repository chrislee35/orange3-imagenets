import os

from AnyQt.QtWidgets import QLabel
from Orange.data import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input

from orangecontrib.imagenets.util.image_loader import load_image, model_input_shape
from orangecontrib.imagenets.util.image_table import image_table_variables
from orangecontrib.imagenets.util.inspect import feature_layers, feature_map_images, filter_images
from orangecontrib.imagenets.widgets.gen_common import pixmap_from_array, show_images


class OWInspectImageNet(widget.OWWidget):
    name = "Inspect ImageNet"
    description = "Show the filters a trained network learned and what each layer sees in an image."
    icon = "icons/inspect.svg"
    priority = 35
    keywords = "filters, feature maps, kernels, layers, visualize"

    class Inputs:
        model = Input("Model", object, auto_summary=False)
        data = Input("Images", Table)

    class Error(widget.OWWidget.Error):
        failed = widget.Msg("{}")

    class Information(widget.OWWidget.Information):
        no_images = widget.Msg("Connect an image table to see what the layers respond to.")

    layer_index = Setting(0)
    image_index = Setting(0)

    want_main_area = True

    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None
        self._layers = []

        box = gui.widgetBox(self.controlArea, "Layer")
        self.layer_combo = gui.comboBox(box, self, "layer_index", callback=self.redraw)
        box = gui.widgetBox(self.controlArea, "Image")
        self.image_spin = gui.spin(box, self, "image_index", 0, 0, label="Row:", callback=self.redraw)
        self.controlArea.layout().addStretch(1)

        self.filter_title = QLabel("<b>Filters</b> (what the layer looks for)")
        self.filter_label = QLabel()
        self.input_title = QLabel("<b>The image</b>")
        self.input_label = QLabel()
        self.map_title = QLabel("<b>Feature maps</b> (where the layer found it; bright = strong response)")
        self.map_label = QLabel()
        for w in (self.input_title, self.input_label, self.filter_title, self.filter_label, self.map_title, self.map_label):
            self.mainArea.layout().addWidget(w)
        self.mainArea.layout().addStretch(1)

    @Inputs.model
    def set_model(self, model):
        self.model = model
        self._layers = feature_layers(model) if model is not None else []
        self.layer_combo.clear()
        self.layer_combo.addItems([label for _, label in self._layers])
        self.layer_index = 0 if self._layers else -1
        self.redraw()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.image_spin.setMaximum(max(len(data) - 1, 0) if data is not None else 0)
        self.image_index = 0
        self.redraw()

    def _image(self):
        origin, col = image_table_variables(self.data)
        path = os.path.join(origin, str(self.data.metas[self.image_index, col]))
        return load_image(path, model_input_shape(self.model))

    def redraw(self):
        self.Error.clear()
        self.Information.no_images.clear()
        for label in (self.filter_label, self.input_label, self.map_label):
            label.clear()
        if self.model is None or not self._layers or self.layer_index < 0:
            return
        index = self._layers[self.layer_index][0]
        layer = self.model.layers[index]
        try:
            if layer.__class__.__name__ == "Conv2D":
                show_images(self.filter_label, filter_images(layer), max_width=420)
            else:
                self.filter_label.setText("(this layer has no filters of its own)")
            if self.data is None:
                self.Information.no_images()
                return
            image = self._image()
            if image is None:
                self.Error.failed("Could not read that image.")
                return
            self.input_label.setPixmap(pixmap_from_array(image.astype("uint8"), max(1, 140 // image.shape[1])))
            show_images(self.map_label, feature_map_images(self.model, index, image), max_width=520)
        except Exception as e:  # shape mismatches, a model with no defined input, ...
            self.Error.failed(str(e))
