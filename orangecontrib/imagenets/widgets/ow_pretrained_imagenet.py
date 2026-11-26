import numpy as np
from Orange.data import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output

from orangecontrib.imagenets.util.pretrained import BASES, build_pretrained


WEIGHTS = "imagenet"  # the tests build the same network without downloading


class OWPretrainedImageNet(widget.OWWidget):
    name = "Pretrained ImageNet"
    description = "A network that already knows how to see (trained on 1.2 million photos), ready to learn your classes."
    icon = "icons/pretrained.svg"
    priority = 5
    keywords = "transfer learning, fine-tune, mobilenet, pretrained"

    class Inputs:
        data = Input("Images", Table)

    class Outputs:
        model = Output("Model", object, auto_summary=False)

    class Error(widget.OWWidget.Error):
        failed = widget.Msg("Could not build the network: {}\n(The first use downloads about 2 MB of weights.)")

    base_index = Setting(0)
    n_classes = Setting(10)
    freeze = Setting(True)

    def __init__(self):
        super().__init__()
        self.data = None
        box = gui.widgetBox(self.controlArea, "Pretrained network")
        gui.comboBox(box, self, "base_index", items=list(BASES), callback=self.commit)
        box = gui.widgetBox(self.controlArea, "Your data")
        self.classes_spin = gui.spin(box, self, "n_classes", 2, 1000, label="Classes:", callback=self.commit)
        gui.checkBox(box, self, "freeze", "Freeze the pretrained part (train only the new last layer)", callback=self.commit)
        self.info_label = gui.label(self.controlArea, self, "")
        self.controlArea.layout().addStretch(1)
        self.commit()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if data is not None and data.domain.class_var is not None and data.domain.class_var.is_discrete:
            self.n_classes = max(2, len(data.domain.class_var.values))
        self.commit()

    def commit(self):
        self.Error.clear()
        label = list(BASES)[self.base_index]
        try:
            model = build_pretrained(label, self.n_classes, self.freeze, weights=WEIGHTS)
        except Exception as e:
            self.Error.failed(str(e))
            self.Outputs.model.send(None)
            return
        trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
        self.info_label.setText(f"{model.count_params():,} weights, {trainable:,} will be trained.\n"
                          f"Images are resized to {model.input_shape[1]} x {model.input_shape[2]} in color.")
        self.Outputs.model.send(model)
