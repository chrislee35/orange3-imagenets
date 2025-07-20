import os
import numpy as np
import cv2

from AnyQt.QtWidgets import QLabel
from PyQt5.QtCore import QThread, pyqtSignal

from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table, DiscreteVariable

class ClassifyWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data

    def run(self):
        domain = self.data.domain
        image_col = None
        origin = None

        for var in domain.metas:
            if var.attributes.get("type") == "image":
                image_col = var
                origin = var.attributes.get("origin")
                break
        image_col_index = domain.metas.index(image_col)
        cv = domain.class_var

        results = []
        total = len(self.data)

        for i, row in enumerate(self.data):
            rel_path = str(row.metas[image_col_index])
            img_path = os.path.join(origin, rel_path)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            pred = self.model.predict(np.expand_dims(img, axis=0), verbose=0)
            pred_class = cv.str_val(np.argmax(pred))
            results.append(pred_class)
            self.progress.emit(int(100 * (i + 1) / total))

        self.finished.emit(results)

class OWImageNetClassify(OWWidget):
    name = "Classify Images"
    description = "Classify images using a trained Keras model."
    icon = "icons/classify.svg"
    priority = 20

    class Inputs:
        data = Input("Data", Table)
        model = Input("Model", object, auto_summary=False)

    class Outputs:
        annotated_data = Output("Annotated Data", Table)

    want_basic_layout = True
    want_control_area = False
    want_main_area = False

    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None

        self.info_label = QLabel("Waiting for input...")
        self.layout().addWidget(self.info_label)

    @Inputs.model
    def set_model(self, model):
        self.model = model
        self.try_classify()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.try_classify()

    def try_classify(self):
        if self.model is not None and self.data is not None:
            self.info_label.setText("Classifying...")
            self.progressBarInit()

            self.worker = ClassifyWorker(self.model, self.data)
            self.worker.progress.connect(self.progressBarSet)
            self.worker.finished.connect(self.handle_results)
            self.worker.start()

    def handle_results(self, predictions):
        var = DiscreteVariable('Prediction', values=[str(i) for i in sorted(set(predictions))])
        annotated = self.data.add_column(var, predictions, to_metas=True)
        self.Outputs.annotated_data.send(annotated)
        self.progressBarFinished()
        self.info_label.setText("Classification complete.")
