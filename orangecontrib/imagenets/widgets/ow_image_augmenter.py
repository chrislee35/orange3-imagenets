from Orange.widgets import widget, settings, gui
from Orange.widgets.widget import Input, Output
from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from AnyQt.QtWidgets import QFileDialog, QVBoxLayout, QLabel, QSpinBox, QCheckBox, QPushButton
from AnyQt.QtCore import Qt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import os
import uuid
import numpy as np
import shutil

class OWImageAugmenter(widget.OWWidget):
    name = "Image Augmenter"
    description = "Applies image augmentation techniques to a set of input images."
    icon = "icons/imageaugmenter.svg"
    priority = 100

    class Inputs:
        images = Input("Image Table", Table)

    class Outputs:
        augmented_images = Output("Augmented Image Table", Table)

    augment_count = settings.Setting(2)
    save_folder = settings.Setting('/home/chris/Downloads/test')
    zoom = settings.Setting(True)
    flip = settings.Setting(True)
    rotate = settings.Setting(True)
    shear = settings.Setting(False)
    brightness = settings.Setting(False)

    def __init__(self):
        super().__init__()

        self.image_table = None
        self.layout_controlArea()
        self.layout_mainArea()

    def layout_controlArea(self):

        # UI
        self.save_folder_button = gui.button(self.controlArea, self, "Select Save Folder", callback=self.select_folder)
        self.folder_label = QLabel(self.save_folder)
        self.controlArea.layout().addWidget(self.folder_label)

        self.augment_spin = QSpinBox()
        self.augment_spin.setMinimum(1)
        self.augment_spin.setValue(self.augment_count)
        self.augment_spin.valueChanged.connect(self.set_augment_count)
        self.controlArea.layout().addWidget(QLabel("Augmentations per Image:"))
        self.controlArea.layout().addWidget(self.augment_spin)

        # Augmentation Options
        self.zoom_cb = QCheckBox("Zoom")
        self.zoom_cb.setChecked(self.zoom)
        self.zoom_cb.stateChanged.connect(lambda: self.set_flag('zoom', self.zoom_cb.isChecked()))
        self.controlArea.layout().addWidget(self.zoom_cb)

        self.flip_cb = QCheckBox("Horizontal Flip")
        self.flip_cb.setChecked(self.flip)
        self.flip_cb.stateChanged.connect(lambda: self.set_flag('flip', self.flip_cb.isChecked()))
        self.controlArea.layout().addWidget(self.flip_cb)

        self.rotate_cb = QCheckBox("Rotation")
        self.rotate_cb.setChecked(self.rotate)
        self.rotate_cb.stateChanged.connect(lambda: self.set_flag('rotate', self.rotate_cb.isChecked()))
        self.controlArea.layout().addWidget(self.rotate_cb)

        self.shear_cb = QCheckBox("Shear")
        self.shear_cb.setChecked(self.shear)
        self.shear_cb.stateChanged.connect(lambda: self.set_flag('shear', self.shear_cb.isChecked()))
        self.controlArea.layout().addWidget(self.shear_cb)

        self.brightness_cb = QCheckBox("Brightness")
        self.brightness_cb.setChecked(self.brightness)
        self.brightness_cb.stateChanged.connect(lambda: self.set_flag('brightness', self.brightness_cb.isChecked()))
        self.controlArea.layout().addWidget(self.brightness_cb)

        self.run_button = QPushButton("Generate Augmented Images")
        self.run_button.clicked.connect(self.generate_augmentations)
        self.controlArea.layout().addWidget(self.run_button)
        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def layout_mainArea(self):
        pass

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Augmented Images")
        if folder:
            self.save_folder = folder
            self.folder_label.setText(folder)

    def set_augment_count(self, val):
        self.augment_count = val

    def set_flag(self, attr, val):
        setattr(self, attr, val)

    @Inputs.images
    def set_data(self, table):
        self.image_table = table

    def generate_augmentations(self):
        if not self.image_table or not self.save_folder:
            self.error("No image data or folder selected.")
            return

        self.progressBarInit()
        domain_attrs = []
        metas = []
        image_col = None
        width_col = None
        height_col = None
        origin = None

        domain = self.image_table.domain

        for var in domain.metas:
            if var.attributes.get("type") == "image":
                image_col = var
                origin = var.attributes.get("origin")
                var.attributes["origin"] = self.save_folder
            elif var.name.lower() == "width":
                width_col = var
            elif var.name.lower() == "height":
                height_col = var
            metas.append(var)

        if not image_col:
            self.error("No image column detected.")
            self.progressBarFinished()
            return

        datagen = ImageDataGenerator(
            zoom_range=0.2 if self.zoom else 0.0,
            horizontal_flip=self.flip,
            rotation_range=30 if self.rotate else 0,
            shear_range=0.2 if self.shear else 0.0,
            brightness_range=(0.7, 1.3) if self.brightness else None,
            fill_mode='nearest')

        new_paths = []
        new_rows = []
        new_y = []
        
        for i, row in enumerate(self.image_table):
            path = row.metas[self.image_table.domain.metas.index(image_col)]
            try:
                img = load_img(os.path.join(origin,path))
            except:
                print("Could not load "+os.path.join(origin,path))
                continue

            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            gen = datagen.flow(x, batch_size=1)

            for j in range(self.augment_count):
                batch = next(gen)
                aug_img = array_to_img(batch[0])
                filename = f"aug_{uuid.uuid4().hex}.png"
                fullpath = os.path.join(self.save_folder, filename)
                aug_img.save(fullpath)
                new_row = list(row.metas)
                new_row[self.image_table.domain.metas.index(image_col)] = filename
                new_rows.append(new_row)
                new_y.append(row.y)
                self.progressBarSet((i * self.augment_count + j) / (len(self.image_table) * self.augment_count) * 100)
        
        metas_array = np.array(new_rows, dtype=object)
        new_table = Table.from_numpy(domain, X=np.empty((len(new_rows), 0)), Y=np.array(new_y), metas=metas_array)
        self.Outputs.augmented_images.send(new_table)
        self.progressBarFinished()

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.imageanalytics.import_images import ImportImages, scan
    import_images = ImportImages()
    data_dir = "/home/chris/Downloads/BilddatenLungenentzuendung/training/krank"
    images = import_images.image_meta(scan(data_dir))
    data, err = import_images(data_dir)
    WidgetPreview(OWImageAugmenter).run(data)