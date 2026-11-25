import tempfile

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel
from Orange.data import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output

from orangecontrib.imagenets.util.image_table import images_to_table
from orangecontrib.imagenets.widgets.gen_common import show_images


class OWGenerateImages(widget.OWWidget):
    name = "Generate Images"
    description = "Draw new images with a trained GAN generator or diffusion model."
    icon = "icons/generate.svg"
    priority = 60
    keywords = "generate, sample, gan, diffusion"

    class Inputs:
        model = Input("Model", object, auto_summary=False)

    class Outputs:
        images = Output("Images", Table)

    class Error(widget.OWWidget.Error):
        failed = widget.Msg("{}")

    count = Setting(16)
    seed = Setting(0)
    class_index = Setting(0)  # 0 = any / not conditioned, else 1 + class number
    guidance = Setting(2.0)
    show_steps = Setting(False)

    def __init__(self):
        super().__init__()
        self.model = None
        self._folder = None

        box = gui.widgetBox(self.controlArea, "Generate")
        gui.spin(box, self, "count", 1, 400, label="Images:")
        gui.spin(box, self, "seed", 0, 10 ** 6, label="Random seed:", tooltip="Same seed, same images.")
        self.class_combo = gui.comboBox(box, self, "class_index", label="Class:", items=["(none)"])
        gui.doubleSpin(box, self, "guidance", 0.0, 10.0, 0.5, label="Guidance:",
                       tooltip="Diffusion only: how strongly to push images toward the chosen class.")
        self.steps_check = gui.checkBox(box, self, "show_steps", "Show denoising steps",
                                        tooltip="Diffusion only: show the first few images as they are denoised.")
        self.button = gui.button(self.controlArea, self, "Generate", callback=self.generate)
        self.controlArea.layout().addStretch()
        self.image_label = QLabel("Connect a Train GAN or Train Diffusion model.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.mainArea.layout().addWidget(self.image_label)
        self._update_controls()

    @Inputs.model
    def set_model(self, model):
        self.model = model
        self.class_combo.clear()
        names = getattr(model, "class_names", None)
        if names:
            first = "(none)"
        elif model is None:
            first = "(none)"
        elif model.kind == "gan":
            first = "(GANs have no classes)"
        else:
            first = "(model was not trained with classes)"
        self.class_combo.addItems([first] + list(names or []))
        self.class_index = 0
        self._update_controls()
        if model is not None:
            self.generate()

    def _update_controls(self):
        diffusion = getattr(self.model, "kind", None) == "diffusion"
        conditioned = diffusion and self.model.class_names is not None
        self.class_combo.setEnabled(conditioned)
        self.steps_check.setEnabled(diffusion)
        self.button.setEnabled(self.model is not None)

    def generate(self):
        self.Error.clear()
        if self.model is None:
            return
        try:
            labels = None
            if self.class_index > 0:
                labels = self.class_index - 1
            names = getattr(self.model, "class_names", None)
            if self.model.kind == "diffusion":
                want_steps = self.show_steps
                result = self.model.sample(self.count, labels=labels, guidance=self.guidance, seed=self.seed,
                                           keep_steps=6 if want_steps else 0)
                images, snaps = result if want_steps else (result, None)
            else:
                images, snaps = self.model.sample(self.count, seed=self.seed), None
        except Exception as e:  # noqa: BLE001 - keep the canvas alive and show what went wrong
            self.Error.failed(str(e))
            return
        if snaps:
            shown = [s[: min(8, len(s))] for s in snaps] + [images[: min(8, len(images))]]
            cols = len(shown[0])
            show_images(self.image_label, [img for row in shown for img in row], cols=cols)
        else:
            show_images(self.image_label, images)
        out_labels = None
        if names is not None:
            out_labels = [names[self.class_index - 1]] * len(images) if self.class_index > 0 else None
        self._folder = tempfile.mkdtemp(prefix="imagenets-generated-")
        self.Outputs.images.send(images_to_table(images, self._folder, out_labels))


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWGenerateImages).run()
