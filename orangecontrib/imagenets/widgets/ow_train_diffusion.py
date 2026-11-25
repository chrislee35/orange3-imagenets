import numpy as np
from AnyQt.QtCore import QCoreApplication, Qt
from AnyQt.QtWidgets import QLabel
from Orange.data import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output

from orangecontrib.imagenets.util.diffusion import train_diffusion
from orangecontrib.imagenets.util.image_loader import load_table_images
from orangecontrib.imagenets.widgets.gen_common import channels, make_loss_plot, shape_controls, show_images


class OWTrainDiffusion(widget.OWWidget):
    name = "Train Diffusion"
    description = "Train a miniature diffusion model that learns to remove noise from images, then draws new ones from pure noise."
    icon = "icons/diffusion.svg"
    priority = 50
    keywords = "diffusion, denoising, stable diffusion, generative"

    class Inputs:
        data = Input("Images", Table)

    class Outputs:
        model = Output("Diffusion Model", object, auto_summary=False)

    class Error(widget.OWWidget.Error):
        no_images = widget.Msg("No images could be read from the table.")
        bad_shape = widget.Msg("{}")

    img_height = Setting(28)
    img_width = Setting(28)
    channels_index = Setting(0)
    epochs = Setting(20)
    batch_size = Setting(64)
    noise_steps = Setting(200)
    base_filters = Setting(16)
    lr_index = Setting(2)
    use_classes = Setting(True)
    preview_every = Setting(5)

    LEARNING_RATES = (1e-4, 5e-4, 1e-3, 2e-3)

    def __init__(self):
        super().__init__()
        self.data = None
        self._stop = False
        self._training = False
        self.losses = []

        shape_controls(self)
        box = gui.widgetBox(self.controlArea, "Training")
        gui.spin(box, self, "epochs", 1, 500, label="Epochs:")
        gui.spin(box, self, "batch_size", 8, 512, label="Batch size:")
        gui.comboBox(box, self, "lr_index", label="Learning rate:", items=[str(r) for r in self.LEARNING_RATES])
        net = gui.widgetBox(self.controlArea, "Model")
        gui.spin(net, self, "noise_steps", 10, 1000, label="Noise steps:", tooltip="How many small steps turn an image into pure noise.")
        gui.spin(net, self, "base_filters", 4, 128, label="Filters:", tooltip="Width of the denoising U-Net.")
        gui.checkBox(net, self, "use_classes", "Condition on the class", tooltip="Tell the model each image's class, like a prompt.")
        gui.spin(net, self, "preview_every", 0, 100, label="Preview every:", tooltip="Epochs between sample drawings (0 = never). Drawing takes a while.")
        self.train_button = gui.button(self.controlArea, self, "Train", callback=self.train)
        self.stop_button = gui.button(self.controlArea, self, "Stop", callback=self._request_stop)
        self.stop_button.setEnabled(False)
        self.controlArea.layout().addStretch()

        self.sample_label = QLabel("Previews appear here every few epochs.")
        self.sample_label.setAlignment(Qt.AlignCenter)
        self.mainArea.layout().addWidget(self.sample_label)
        self.plot = make_loss_plot("Noise prediction error")
        self.loss_curve = self.plot.plot(pen="r", name="Loss")
        self.mainArea.layout().addWidget(self.plot)

    @Inputs.data
    def set_data(self, data):
        self.data = data

    def _request_stop(self):
        self._stop = True

    def _on_epoch(self, epoch, loss, model):
        self.losses.append(loss)
        self.loss_curve.setData(np.arange(1, len(self.losses) + 1), np.array(self.losses))
        self.progressBarSet(100 * epoch / self.epochs)
        QCoreApplication.processEvents()
        if self.preview_every and (epoch % self.preview_every == 0 or epoch == self.epochs):
            labels = list(range(len(model.class_names))) if model.class_names else None
            n = min(len(labels), 10) if labels else 8
            images = model.sample(n, labels=labels[:n] if labels else None, seed=0)
            show_images(self.sample_label, images, cols=n)
            QCoreApplication.processEvents()

    def train(self):
        self.Error.clear()
        if self.data is None or self._training:
            return
        shape = (self.img_height, self.img_width, channels(self))
        self.progressBarInit()
        X, labels = load_table_images(self.data, shape)
        if len(X) == 0:
            self.Error.no_images()
            self.progressBarFinished()
            return
        self.losses = []
        self._stop, self._training = False, True
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        try:
            model = train_diffusion(X, labels if self.use_classes else None, self.epochs, self.batch_size, self.noise_steps,
                                    self.base_filters, self.LEARNING_RATES[self.lr_index], on_epoch=self._on_epoch,
                                    should_stop=lambda: self._stop)
        except ValueError as e:
            self.Error.bad_shape(str(e))
            model = None
        finally:
            self._training = False
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progressBarFinished()
        self.Outputs.model.send(model)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWTrainDiffusion).run()
