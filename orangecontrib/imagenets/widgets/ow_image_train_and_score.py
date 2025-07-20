from Orange.widgets import widget
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Output, Input
from Orange.data import Table

from PyQt5.QtWidgets import QLabel, QPushButton, QSpinBox, QComboBox
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QFont

import os
import cv2
import numpy as np
from pyqtgraph import PlotWidget, PlotCurveItem, ScatterPlotItem

from keras.utils import to_categorical
from keras.models import clone_model
from keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder

class KerasCallback(Callback):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.widget.loss_values.append(logs.get('loss', 0))
        self.widget.accuracy_values.append(logs.get('accuracy', 0))

        epochs = list(range(1, len(self.widget.loss_values) + 1))
        self.widget.loss_curve.setData(epochs, self.widget.loss_values)
        self.widget.accuracy_curve.setData(epochs, self.widget.accuracy_values)
        self.widget.update_tooltips(self.widget.loss_scatter, self.widget.loss_values, "Loss")
        self.widget.update_tooltips(self.widget.accuracy_scatter, self.widget.accuracy_values, "Accuracy")

        self.widget.progressBarSet(100 * (epoch + 1) / self.widget.epochs)
        QCoreApplication.processEvents()

class OWImageTrainAndScore(widget.OWWidget):
    name = "Image Train and Score"
    description = "Train a Keras model on image data and show live performance."
    icon = "icons/train_score.svg"
    priority = 10

    class Inputs:
        model = Input("Learner", object)
        data = Input("Evaluation Data", Table)

    class Outputs:
        trained_model = Output("Trained Model", object)

    batch_size = Setting(32)
    dropout_rate = Setting(0.5)
    epochs = Setting(10)

    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None

        self.loss_values = []
        self.accuracy_values = []

        self.init_controls()
        self.setup_training_graph()

    def init_controls(self):
        self.controlArea.layout().addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(self.batch_size)
        self.batch_size_spin.setToolTip("Number of samples per training batch.")
        self.batch_size_spin.valueChanged.connect(self._on_batch_size_changed)
        self.controlArea.layout().addWidget(self.batch_size_spin)

        self.controlArea.layout().addWidget(QLabel("Dropout Rate:"))
        self.dropout_box = QComboBox()
        self.dropout_box.addItems(["0.0", "0.25", "0.5", "0.75"])
        self.dropout_box.setToolTip("Proportion of dropout to apply to the model.")
        self.dropout_box.setCurrentText(str(self.dropout_rate))
        self.dropout_box.currentTextChanged.connect(self._on_dropout_changed)
        self.controlArea.layout().addWidget(self.dropout_box)

        self.controlArea.layout().addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 512)
        self.epochs_spin.setValue(self.epochs)
        self.epochs_spin.setToolTip("Number of epochs of training .")
        self.epochs_spin.valueChanged.connect(self._on_epochs_changed)
        self.controlArea.layout().addWidget(self.epochs_spin)        

        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.train)
        self.controlArea.layout().addWidget(self.train_button)

        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def setup_training_graph(self):
        self.graph = PlotWidget(title="Training Progress")
        self.graph.setLabel('left', 'Value')
        self.graph.setLabel('bottom', 'Epoch')
        self.graph.showGrid(x=True, y=True)
        palette = self.palette()
        bg_color = palette.window().color().name()

        self.graph.setBackground(bg_color)
        self.graph.addLegend()

        font = QFont()
        font.setPointSize(10)
        self.graph.getAxis('bottom').setTickFont(font)
        self.graph.getAxis('left').setTickFont(font)

        self.loss_curve = PlotCurveItem(pen='r', name="Loss")
        self.accuracy_curve = PlotCurveItem(pen='g', name="Accuracy")

        self.loss_scatter = ScatterPlotItem(pen=None, symbol='o', brush='r', size=6)
        self.accuracy_scatter = ScatterPlotItem(pen=None, symbol='t', brush='g', size=6)

        self.graph.addItem(self.loss_curve, "Loss")
        self.graph.addItem(self.accuracy_curve, "Accuracy")
        self.graph.addItem(self.loss_scatter)
        self.graph.addItem(self.accuracy_scatter)

        self.mainArea.layout().addWidget(self.graph)

    def update_tooltips(self, scatter_item, values, label):
        spots = [{
            'pos': (i + 1, v),
            'data': f"{label} - Epoch {i + 1}\nValue: {v:.4f}",
            'brush': scatter_item.opts['brush'],
        } for i, v in enumerate(values)]
        scatter_item.setData(spots)

        def on_hover(spots):
            if spots:
                self.graph.setToolTip(spots[0].data())
            else:
                self.graph.setToolTip("")
        scatter_item.sigHovered.connect(on_hover)

    @Inputs.model
    def set_model(self, model):
        self.model = model
        if self.model and self.data:
            self.train()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if self.model and self.data:
            self.train()

    def _on_batch_size_changed(self, value):
        self.batch_size = value

    def _on_dropout_changed(self, value):
        self.dropout_rate = float(value)

    def _on_epochs_changed(self, value):
        self.epochs = int(value)

    def prepare_data(self):
        X = []
        y = []

        domain = self.data.domain
        image_col = None
        origin = None

        for var in domain.metas:
            if var.attributes.get("type") == "image":
                image_col = var
                origin = var.attributes.get("origin")
                break
        image_col_index = domain.metas.index(image_col)

        for row in self.data:
            path = row.metas[image_col_index]
            img_path = os.path.join(origin, path)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            X.append(img)
            y.append(str(row.get_class()))

        X = np.array(X)
        le = LabelEncoder()
        y_int = le.fit_transform(y)
        y_cat = to_categorical(y_int)
        return X, y_cat, le

    def train(self):
        if self.model is None or self.data is None:
            self.error("Missing model or data.")
            return

        model = clone_model(self.model)
        model.set_weights(self.model.get_weights())

        self.loss_values.clear()
        self.accuracy_values.clear()
        self.progressBarInit()

        X, y, le = self.prepare_data()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            callbacks=[KerasCallback(self)]
        )

        model.class_names = le.classes_.tolist()

        self.progressBarFinished()
        self.Outputs.trained_model.send(model)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    # from orangecontrib.imageanalytics.import_images import ImportImages, scan
    # import_images = ImportImages()
    # data_dir = "/home/chris/Downloads/BilddatenLungenentzuendung/training/krank"
    # images = import_images.image_meta(scan(data_dir))
    # data, err = import_images(data_dir)
    WidgetPreview(OWImageTrainAndScore).run()