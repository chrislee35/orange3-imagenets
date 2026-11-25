import numpy as np
from AnyQt.QtCore import QCoreApplication, Qt
from AnyQt.QtWidgets import QLabel
from Orange.data import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output

from orangecontrib.imagenets.util.gan import train_gan
from orangecontrib.imagenets.util.image_loader import load_table_images
from orangecontrib.imagenets.widgets.gen_common import channels, make_loss_plot, shape_controls, show_images


class OWTrainGAN(widget.OWWidget):
    name = "Train GAN"
    description = "Train a small generative adversarial network (GAN) that learns to draw images like the ones it is shown."
    icon = "icons/gan.svg"
    priority = 40
    keywords = "gan, generative, generator, discriminator"

    class Inputs:
        data = Input("Images", Table)

    class Outputs:
        generator = Output("Generator", object, auto_summary=False)

    class Error(widget.OWWidget.Error):
        no_images = widget.Msg("No images could be read from the table.")
        bad_shape = widget.Msg("{}")

    img_height = Setting(28)
    img_width = Setting(28)
    channels_index = Setting(0)
    epochs = Setting(15)
    batch_size = Setting(64)
    latent_dim = Setting(64)
    base_filters = Setting(32)
    lr_index = Setting(1)

    LEARNING_RATES = (1e-4, 2e-4, 5e-4, 1e-3)

    def __init__(self):
        super().__init__()
        self.data = None
        self._stop = False
        self._training = False
        self.d_losses, self.g_losses, self.d_accs = [], [], []

        shape_controls(self)
        box = gui.widgetBox(self.controlArea, "Training")
        gui.spin(box, self, "epochs", 1, 500, label="Epochs:")
        gui.spin(box, self, "batch_size", 8, 512, label="Batch size:")
        gui.comboBox(box, self, "lr_index", label="Learning rate:", items=[str(r) for r in self.LEARNING_RATES])
        net = gui.widgetBox(self.controlArea, "Networks")
        gui.spin(net, self, "latent_dim", 2, 512, label="Noise size:", tooltip="Length of the random vector the generator starts from.")
        gui.spin(net, self, "base_filters", 4, 128, label="Filters:", tooltip="Width of the generator and discriminator.")
        self.train_button = gui.button(self.controlArea, self, "Train", callback=self.train)
        self.stop_button = gui.button(self.controlArea, self, "Stop", callback=self._request_stop)
        self.stop_button.setEnabled(False)
        self.controlArea.layout().addStretch()

        self.sample_label = QLabel("Generated samples appear here after each epoch.")
        self.sample_label.setAlignment(Qt.AlignCenter)
        self.mainArea.layout().addWidget(self.sample_label)
        self.plot = make_loss_plot("Discriminator vs generator")
        self.d_curve = self.plot.plot(pen="c", name="Discriminator loss")
        self.g_curve = self.plot.plot(pen="m", name="Generator loss")
        self.acc_curve = self.plot.plot(pen="g", name="Discriminator accuracy")
        self.mainArea.layout().addWidget(self.plot)

    @Inputs.data
    def set_data(self, data):
        self.data = data

    def _request_stop(self):
        self._stop = True

    def _on_epoch(self, epoch, d_loss, g_loss, d_acc, samples):
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        self.d_accs.append(d_acc)
        x = np.arange(1, len(self.d_losses) + 1)
        self.d_curve.setData(x, np.array(self.d_losses))
        self.g_curve.setData(x, np.array(self.g_losses))
        self.acc_curve.setData(x, np.array(self.d_accs))
        show_images(self.sample_label, samples, cols=8)
        self.progressBarSet(100 * epoch / self.epochs)
        QCoreApplication.processEvents()

    def train(self):
        self.Error.clear()
        if self.data is None or self._training:
            return
        shape = (self.img_height, self.img_width, channels(self))
        self.progressBarInit()
        X, _ = load_table_images(self.data, shape)
        if len(X) == 0:
            self.Error.no_images()
            self.progressBarFinished()
            return
        self.d_losses, self.g_losses, self.d_accs = [], [], []
        self._stop, self._training = False, True
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        try:
            model = train_gan(X, self.epochs, self.batch_size, self.latent_dim, self.base_filters, self.LEARNING_RATES[self.lr_index],
                              on_epoch=self._on_epoch, should_stop=lambda: self._stop)
        except ValueError as e:
            self.Error.bad_shape(str(e))
            model = None
        finally:
            self._training = False
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progressBarFinished()
        self.Outputs.generator.send(model)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWTrainGAN).run()
