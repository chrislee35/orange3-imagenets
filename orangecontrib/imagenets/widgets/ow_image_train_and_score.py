from Orange.widgets import widget
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Output, Input
from Orange.data import Table

from PyQt5.QtWidgets import QLabel, QPushButton, QSpinBox
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QFont

import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotCurveItem, ScatterPlotItem

from keras.utils import to_categorical
from keras.models import clone_model
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from orangecontrib.imagenets.util.image_loader import load_image, model_input_shape

class KerasCallback(Callback):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.widget.loss_values.append(logs.get('loss', 0))
        self.widget.accuracy_values.append(logs.get('accuracy', 0))
        if 'val_loss' in logs:
            self.widget.val_loss_values.append(logs['val_loss'])
            self.widget.val_accuracy_values.append(logs.get('val_accuracy', 0))
            self.widget.val_loss_curve.setData(np.arange(1, len(self.widget.val_loss_values) + 1), np.array(self.widget.val_loss_values))
            self.widget.val_accuracy_curve.setData(np.arange(1, len(self.widget.val_accuracy_values) + 1), np.array(self.widget.val_accuracy_values))

        epochs = list(range(1, len(self.widget.loss_values) + 1))
        self.widget.loss_curve.setData(epochs, self.widget.loss_values)
        self.widget.accuracy_curve.setData(epochs, self.widget.accuracy_values)
        self.widget.update_tooltips(self.widget.loss_scatter, self.widget.loss_values, "Loss")
        self.widget.update_tooltips(self.widget.accuracy_scatter, self.widget.accuracy_values, "Accuracy")

        self.widget.progressBarSet(100 * (epoch + 1) / self.widget.epochs)
        QCoreApplication.processEvents()

class OWImageTrainAndScore(widget.OWWidget):
    name = "Train ImageNet"
    description = "Train a Keras model on image data and show live performance."
    icon = "icons/train.svg"
    priority = 10

    class Inputs:
        model = Input("Learner", object, auto_summary=False)
        data = Input("Evaluation Data", Table)

    class Outputs:
        trained_model = Output("Trained Model", object, auto_summary=False)

    batch_size = Setting(32)
    epochs = Setting(10)
    validation_percent = Setting(0)  # share of the images held out to score the model after each epoch

    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None

        self.loss_values = []
        self.accuracy_values = []
        self.val_loss_values = []
        self.val_accuracy_values = []

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

        self.controlArea.layout().addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 512)
        self.epochs_spin.setValue(self.epochs)
        self.epochs_spin.setToolTip("Number of epochs of training .")
        self.epochs_spin.valueChanged.connect(self._on_epochs_changed)
        self.controlArea.layout().addWidget(self.epochs_spin)        

        self.controlArea.layout().addWidget(QLabel("Validation split (%):"))
        self.validation_spin = QSpinBox()
        self.validation_spin.setRange(0, 50)
        self.validation_spin.setValue(self.validation_percent)
        self.validation_spin.setToolTip("Percent of the images held out and scored after every epoch (0 = none).")
        self.validation_spin.valueChanged.connect(self._on_validation_changed)
        self.controlArea.layout().addWidget(self.validation_spin)

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
        self.graph.addLegend(offset=(-20, 150))  # right side, clear of the curves

        font = QFont()
        font.setPointSize(10)
        self.graph.getAxis('bottom').setTickFont(font)
        self.graph.getAxis('left').setTickFont(font)

        self.loss_curve = PlotCurveItem(pen='r', name="Loss")
        self.accuracy_curve = PlotCurveItem(pen='g', name="Accuracy")

        self.val_loss_curve = PlotCurveItem(pen=pg.mkPen('r', style=Qt.DashLine), name="Validation loss")
        self.val_accuracy_curve = PlotCurveItem(pen=pg.mkPen('g', style=Qt.DashLine), name="Validation accuracy")
        self.graph.addItem(self.val_loss_curve, "Validation loss")
        self.graph.addItem(self.val_accuracy_curve, "Validation accuracy")

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

    def _on_epochs_changed(self, value):
        self.epochs = int(value)

    def _on_validation_changed(self, value):
        self.validation_percent = int(value)

    def prepare_data(self):
        self.input_shape = model_input_shape(self.model)
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
            img = load_image(img_path, self.input_shape)
            if img is None:
                continue
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
        self.val_loss_values.clear()
        self.val_accuracy_values.clear()
        self.val_loss_curve.setData(np.array([]), np.array([]))
        self.val_accuracy_curve.setData(np.array([]), np.array([]))
        self.progressBarInit()

        X, y, le = self.prepare_data()
        n_classes, n_outputs = y.shape[1], model.output_shape[-1]
        if n_classes != n_outputs:
            self.progressBarFinished()
            self.error(f"The data has {n_classes} classes but the model's last layer has {n_outputs} units. "
                       "Set the last Dense layer's units to the number of classes.")
            return
        self.error()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Hold out a random, class-balanced share. (Keras's own validation_split takes the LAST rows, and image folders
        # are sorted by class, so it would hold out whole classes.)
        validation_data = None
        if self.validation_percent:
            n_val = max(round(len(X) * self.validation_percent / 100), n_classes)  # stratifying needs one per class
            if n_val >= len(X):
                self.progressBarFinished()
                self.error("Too few images for that validation split.")
                return
            X, X_val, y, y_val = train_test_split(X, y, test_size=n_val, stratify=y.argmax(axis=1), random_state=0)
            validation_data = (X_val, y_val)

        model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
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