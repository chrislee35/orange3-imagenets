from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Output, Input
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable, StringVariable

from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QSpinBox, QComboBox
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QFont

import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotCurveItem, ScatterPlotItem, LegendItem

class OWImageTrainAndScore(widget.OWWidget):
    name = "Train ImageNet"
    description = "Train an ImageNet model and display learning curves."
    icon = "icons/train.svg"
    priority = 20

    class Inputs:
        model = Input("Learner", object)
        data = Input("Evaluation Data", Table)

    class Outputs:
        trained_model = Output("Trained Model", object)

    batch_size = Setting(32)
    dropout_rate = Setting(0.5)

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

    def train(self):
        if not self.model or not self.data:
            self.error("Missing model or data")
            return

        self.progressBarInit()
        self.loss_values = []
        self.accuracy_values = []

        for epoch in range(10):
            # Simulated training
            self.loss_values.append(np.exp(-0.3 * epoch) + np.random.rand() * 0.1)
            self.accuracy_values.append(1 - np.exp(-0.3 * epoch) + np.random.rand() * 0.05)

            epochs = list(range(1, len(self.loss_values) + 1))
            self.loss_curve.setData(epochs, self.loss_values)
            self.accuracy_curve.setData(epochs, self.accuracy_values)
            self.update_tooltips(self.loss_scatter, self.loss_values, "Loss")
            self.update_tooltips(self.accuracy_scatter, self.accuracy_values, "Accuracy")

            self.progressBarSet(100 * (epoch + 1) / 10)
            QCoreApplication.processEvents()

        self.progressBarFinished()
        self.Outputs.trained_model.send(self.model)

    @Inputs.model
    def set_model(self, model):
        self.model = model

    @Inputs.data
    def set_data(self, data):
        self.data = data

    def _on_batch_size_changed(self, value):
        self.batch_size = value

    def _on_dropout_changed(self, value):
        self.dropout_rate = float(value)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    # from orangecontrib.imageanalytics.import_images import ImportImages, scan
    # import_images = ImportImages()
    # data_dir = "/home/chris/Downloads/BilddatenLungenentzuendung/training/krank"
    # images = import_images.image_meta(scan(data_dir))
    # data, err = import_images(data_dir)
    WidgetPreview(OWImageTrainAndScore).run()