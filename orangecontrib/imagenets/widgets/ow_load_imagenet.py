import os
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFileDialog, QLabel
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Output
from keras.models import model_from_json, load_model

class OWLoadKerasModel(OWWidget):
    name = "Load ImageNet"
    description = "Load a Keras Sequential model from .h5 or JSON+weights."
    icon = "icons/load.svg"
    priority = 10

    class Outputs:
        model = Output("Model", object, auto_summary=False)

    last_dir = Setting(os.path.expanduser("~"))
    load_file = Setting("")
    load_type = Setting("")
    
    want_control_area = False

    def __init__(self):
        super().__init__()
        self.model = None
        self.mainArea.layout().addWidget(QLabel("Load Options:"))
        gui.button(self.mainArea, self, "Load from H5", callback=self.load_h5_dialog, width=250)
        gui.button(self.mainArea, self, "Load from JSON + Weights", callback=self.load_json_dialog, width=250)
        self.mainArea.layout().setAlignment(Qt.AlignTop)
        self.adjustSize()

        if len(self.load_file) > 0:
            if self.load_type == 'h5':
                self.load_h5(self.load_file)
            elif self.load_type == 'json':
                self.load_json(self.load_file)
        
    def load_h5_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Model from H5", self.last_dir, "H5 Files (*.h5)")
        if filename:
            self.load_h5(filename)

    def load_h5(self, filename):
        self.last_dir = os.path.dirname(filename)
        self.load_file = filename
        self.load_type = 'h5'
        try:
            self.model = load_model(filename)
            self.Outputs.model.send(self.model)
            #self.setStatusMessage("Loaded model from: " + os.path.basename(filename))
        except Exception as e:
            self.error(str(e))

    def load_json_dialog(self):
        json_path, _ = QFileDialog.getOpenFileName(self, "Load Model Architecture", self.last_dir, "JSON Files (*.json)")
        if json_path:
            self.load_json(json_path)

    def load_json(self, json_path):
        self.last_dir = os.path.dirname(json_path)
        weights_path = os.path.splitext(json_path)[0] + "_weights.h5"
        self.load_file = json_path
        self.load_type = 'json'
        try:
            with open(json_path, "r") as f:
                json_str = f.read()
                self.model = model_from_json(json_str)
            self.model.load_weights(weights_path)
            self.Outputs.model.send(self.model)
            #self.setStatusMessage(f"Loaded: {os.path.basename(json_path)} + weights")
        except Exception as e:
            self.error(str(e))

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWLoadKerasModel).run()

    