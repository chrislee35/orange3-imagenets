import os
from AnyQt.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input
from Orange.widgets.settings import Setting


class OWSaveImageNet(OWWidget):
    name = "Save ImageNet"
    description = "Save a trained Keras Sequential model, an ImageNet, to .h5 or .json format."
    icon = "icons/save.svg"
    priority = 20

    class Inputs:
        model = Input("Model", object, auto_summary=False)

    last_dir = Setting(os.path.expanduser("~"))
    want_control_area = False

    def __init__(self):
        super().__init__()
        self.model = None

        gui.button(self.mainArea, self, "Save as .h5", callback=self.save_as_h5, width=250)
        gui.button(self.mainArea, self, "Save as JSON", callback=self.save_as_json, width=250)
        self.mainArea.layout().setAlignment(Qt.AlignTop)
        self.adjustSize()

    @Inputs.model
    def set_model(self, model):
        self.model = model

    def save_as_h5(self):
        if self.model is None:
            self.error("No model to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Keras Model", self.last_dir, "HDF5 files (*.h5)")
        if filename:
            if not filename.endswith(".h5"):
                filename += ".h5"
            self.model.save(filename)
            self.last_dir = os.path.dirname(filename)

    def save_as_json(self):
        if self.model is None:
            self.error("No model to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Keras Model Architecture", self.last_dir, "JSON files (*.json)")
        if filename:
            if not filename.endswith(".json"):
                filename += ".json"
            with open(filename, "w") as f:
                f.write(self.model.to_json(indent=2))
            self.last_dir = os.path.dirname(filename)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWSaveImageNet).run()