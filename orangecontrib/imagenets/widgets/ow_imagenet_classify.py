import os
import numpy as np

from AnyQt.QtWidgets import QLabel
from PyQt5.QtCore import QThread, pyqtSignal

from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table, DiscreteVariable

from orangecontrib.imagenets.util.image_loader import load_image, model_input_shape

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
        shape = model_input_shape(self.model)
        class_names = getattr(self.model, "class_names", None)
        cv = domain.class_var

        results = []
        batch, batch_size = [], 64
        total = len(self.data)

        def flush():
            if batch:
                for pred in np.argmax(self.model.predict(np.stack(batch), verbose=0), axis=1):
                    results.append(class_names[pred] if class_names else cv.str_val(pred))
                batch.clear()

        for i, row in enumerate(self.data):
            img = load_image(os.path.join(origin, str(row.metas[image_col_index])), shape)
            if img is None:
                continue
            batch.append(img)
            if len(batch) == batch_size:
                flush()
            self.progress.emit(int(100 * (i + 1) / total))
        flush()

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
        text = "Classification complete."
        self.accuracy = None
        if self.data.domain.class_var is not None and len(predictions) == len(self.data):
            truth = [str(self.data.domain.class_var.values[int(v)]) for v in self.data.Y]
            self.accuracy = sum(t == str(p) for t, p in zip(truth, predictions)) / len(truth)
            text = f"Classified {len(truth)} images. Accuracy: {self.accuracy:.1%}"
        self.info_label.setText(text)
