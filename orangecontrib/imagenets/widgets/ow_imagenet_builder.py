from AnyQt.QtWidgets import (
    QComboBox, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
    QWidget, QScrollArea, QFrame, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit, QMessageBox
)
from AnyQt.QtCore import Qt
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.settings import Setting
import keras.layers as klayers
import keras.models as kmodels
import json

PREBUILT_MODELS = {
    "None": [],
    "SimpleCNN": [
        {"type": "Conv2D", "filters": 32, "kernel_size": 3, "activation": "relu"},
        {"type": "MaxPooling2D", "pool_size": 2},
        {"type": "Flatten"},
        {"type": "Dense", "units": 10, "activation": "softmax"}
    ],
    "LeNet": [
        {"type": "Conv2D", "filters": 6, "kernel_size": 5, "activation": "tanh"},
        {"type": "AveragePooling2D", "pool_size": 2},
        {"type": "Conv2D", "filters": 16, "kernel_size": 5, "activation": "tanh"},
        {"type": "AveragePooling2D", "pool_size": 2},
        {"type": "Flatten"},
        {"type": "Dense", "units": 120, "activation": "tanh"},
        {"type": "Dense", "units": 84, "activation": "tanh"},
        {"type": "Dense", "units": 10, "activation": "softmax"}
    ],
    "MiniResNet": [
        {"type": "Conv2D", "filters": 64, "kernel_size": 3, "padding": "same", "activation": "relu"},
        {"type": "BatchNormalization"},
        {"type": "Conv2D", "filters": 64, "kernel_size": 3, "padding": "same", "activation": "relu"},
        {"type": "BatchNormalization"},
        {"type": "MaxPooling2D", "pool_size": 2},
        {"type": "Conv2D", "filters": 128, "kernel_size": 3, "padding": "same", "activation": "relu"},
        {"type": "GlobalAveragePooling2D"},
        {"type": "Dense", "units": 10, "activation": "softmax"}
    ]
}

class OWImageNetBuilder(OWWidget):
    name = "ImageNet Builder"
    description = "Build a custom image neural network with Keras"
    icon = "icons/imagenet_builder.svg"
    priority = 10

    model_config = Setting("[]")  # JSON list of layers

    class Outputs:
        model = Output("Model", object)

    def __init__(self):
        super().__init__()
        self.model_layers = []

        self._init_controls()
        self._init_main_area()
        self._load_saved_config()

    def _init_controls(self):
        export_box = gui.widgetBox(self.controlArea, "Export Model")
        export_h5_btn = QPushButton("Export to .h5")
        export_h5_btn.clicked.connect(self._export_model_h5)
        export_box.layout().addWidget(export_h5_btn)

        export_json_btn = QPushButton("Export to JSON")
        export_json_btn.clicked.connect(self._export_model_json)
        export_box.layout().addWidget(export_json_btn)

        box = gui.widgetBox(self.controlArea, "Prebuilt ImageNets")

        self.prebuilt_combo = QComboBox()
        self.prebuilt_combo.addItems(PREBUILT_MODELS.keys())
        self.prebuilt_combo.currentTextChanged.connect(self.load_prebuilt_model)
        box.layout().addWidget(self.prebuilt_combo)
        box.layout().setAlignment(Qt.AlignTop)

        layer_box = gui.widgetBox(self.controlArea, "Add Layers")
        for layer_type in ["ZeroPadding2D", "Conv2D", "BatchNormalization", "Activation",
                           "MaxPooling2D", "Add", "GlobalAveragePooling2D", "Dropout", "Dense"]:
            btn = QPushButton(layer_type)
            btn.setToolTip(f"Add a {layer_type} layer")
            btn.clicked.connect(lambda checked, l=layer_type: self.add_layer(l))
            layer_box.layout().addWidget(btn)
        layer_box.layout().setAlignment(Qt.AlignTop)
        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def _init_main_area(self):
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.main_widget.layout().setAlignment(Qt.AlignTop)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_layers)
        self.main_layout.addWidget(clear_btn)

        self.layer_container = QVBoxLayout()
        self.main_layout.addLayout(self.layer_container)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container.setLayout(self.main_layout)
        scroll.setWidget(container)
        self.mainArea.layout().addWidget(scroll)
        self.mainArea.layout().setAlignment(Qt.AlignTop)

    def clear_layers(self):
        while self.layer_container.count():
            item = self.layer_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.model_layers = []
        self.add_layer("Rescaling", insert_at_top=True)
        self.add_layer("Dense", default=True)

    def add_layer(self, layer_type, insert_at_top=False, default=False):
        config = self.default_config(layer_type)
        if default and layer_type == "Dense":
            config = {"type": "Dense", "units": 10, "activation": "softmax"}

        insert_index = 0 if insert_at_top else (len(self.model_layers) - 1 if self.model_layers else 0)
        self.model_layers.insert(insert_index, config)

        self._rebuild_ui()
        self._update_model_config()

    def _rebuild_ui(self):
        while self.layer_container.count():
            item = self.layer_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for i, layer_config in enumerate(self.model_layers):
            widget = self._build_layer_widget(layer_config, i)
            self.layer_container.addWidget(widget)

    def _build_layer_widget(self, config, index):
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        layout = QFormLayout()
        frame.setLayout(layout)

        layout.addRow(QLabel(f"Layer: {config['type']}"))

        for key, val in config.items():
            if key == "type":
                continue
            tooltip = f"{key} parameter for {config['type']} layer"
            if isinstance(val, int):
                spin = QSpinBox()
                spin.setMaximum(1024)
                spin.setValue(val)
                spin.setToolTip(tooltip)
                spin.valueChanged.connect(lambda v, k=key: self._update_param(config, k, v))
                layout.addRow(key, spin)
            elif isinstance(val, float):
                dspin = QDoubleSpinBox()
                dspin.setValue(val)
                dspin.setSingleStep(0.1)
                dspin.setToolTip(tooltip)
                dspin.valueChanged.connect(lambda v, k=key: self._update_param(config, k, v))
                layout.addRow(key, dspin)
            elif isinstance(val, str):
                line = QLineEdit(val)
                line.setToolTip(tooltip)
                line.textChanged.connect(lambda v, k=key: self._update_param(config, k, v))
                layout.addRow(key, line)

        btn_layout = QHBoxLayout()
        btn_up = QPushButton("↑")
        btn_up.setToolTip("Move layer up")
        btn_up.clicked.connect(lambda: self._move_layer(index, -1))
        btn_up.setMaximumWidth(28)
        btn_down = QPushButton("↓")
        btn_down.setToolTip("Move layer down")
        btn_down.clicked.connect(lambda: self._move_layer(index, 1))
        btn_down.setMaximumWidth(28)
        btn_delete = QPushButton("✕")
        btn_delete.setToolTip("Delete this layer")
        btn_delete.clicked.connect(lambda: self._delete_layer(index))
        btn_delete.setMaximumWidth(28)

        btn_layout.addWidget(btn_up)
        btn_layout.addWidget(btn_down)
        btn_layout.addWidget(btn_delete)
        btn_layout.layout().setAlignment(Qt.AlignRight)
        layout.addRow(btn_layout)

        return frame

    def _move_layer(self, index, direction):
        new_index = index + direction
        if 0 <= new_index < len(self.model_layers):
            self.model_layers[index], self.model_layers[new_index] = self.model_layers[new_index], self.model_layers[index]
            self._rebuild_ui()
            self._update_model_config()

    def _delete_layer(self, index):
        if len(self.model_layers) <= 2:
            QMessageBox.warning(self, "Cannot delete", "Network must have at least a Rescaling and Dense layer.")
            return
        del self.model_layers[index]
        self._rebuild_ui()
        self._update_model_config()

    def _update_param(self, config, key, value):
        config[key] = value
        self._update_model_config()

    def _update_model_config(self):
        self.model_config = json.dumps(self.model_layers)
        try:
            model = self._build_keras_model()
            self.Outputs.model.send(model)
        except Exception as e:
            print(f"Model build failed: {e}")
            self.Outputs.model.send(None)

    def _build_keras_model(self):
        model = kmodels.Sequential()
        for layer_cfg in self.model_layers:
            config = dict(layer_cfg)
            layer_type = config.pop("type")
            layer_class = getattr(klayers, layer_type)
            model.add(layer_class(**config))
        return model

    def _load_saved_config(self):
        try:
            self.model_layers = json.loads(self.model_config)
            self._rebuild_ui()
            self._update_model_config()
        except Exception as e:
            print(f"Failed to load saved config: {e}")
            self.clear_layers()

    def _export_model_h5(self):
        try:
            model = self._build_keras_model()
            model.save("custom_model.h5")
            QMessageBox.information(self, "Success", "Model saved to custom_model.h5")
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", str(e))

    def _export_model_json(self):
        try:
            model = self._build_keras_model()
            with open("custom_model.json", "w") as f:
                f.write(model.to_json())
            QMessageBox.information(self, "Success", "Model structure saved to custom_model.json")
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", str(e))

    def load_prebuilt_model(self, name):
        if name in PREBUILT_MODELS:
            self.model_layers = PREBUILT_MODELS[name].copy()
            self._rebuild_ui()
            self._update_model_config()

    def default_config(self, layer_type):
        defaults = {
            "ZeroPadding2D": {"type": "ZeroPadding2D", "padding": 1},
            "Conv2D": {"type": "Conv2D", "filters": 32, "kernel_size": 3, "activation": "relu"},
            "BatchNormalization": {"type": "BatchNormalization"},
            "Activation": {"type": "Activation", "activation": "relu"},
            "MaxPooling2D": {"type": "MaxPooling2D", "pool_size": 2},
            "Add": {"type": "Add"},
            "GlobalAveragePooling2D": {"type": "GlobalAveragePooling2D"},
            "Dropout": {"type": "Dropout", "rate": 0.5},
            "Dense": {"type": "Dense", "units": 64, "activation": "relu"},
            "Rescaling": {"type": "Rescaling", "scale": 1./255}
        }
        return defaults.get(layer_type, {"type": layer_type})

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWImageNetBuilder).run()