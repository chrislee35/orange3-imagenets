from Orange.widgets import widget, settings, gui
from Orange.widgets.widget import Input, Output
from Orange.data import Table
from AnyQt.QtWidgets import QFileDialog, QVBoxLayout, QLabel, QSpinBox, QCheckBox, QPushButton, QGroupBox
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QPixmap, QFont
from PIL import Image, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import os
import uuid
import numpy as np


class OWImageAugmenter(widget.OWWidget):
    name = "Augment Images"
    description = "Applies image augmentation techniques to a set of input images."
    icon = "icons/augment.svg"
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
    blur = settings.Setting(False)
    gaussian_noise = settings.Setting(False)


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

        self.blur_cb = QCheckBox("Blur")
        self.blur_cb.setChecked(self.blur)
        self.blur_cb.stateChanged.connect(lambda: self.set_flag('blur', self.blur_cb.isChecked()))
        self.controlArea.layout().addWidget(self.blur_cb)

        self.gauss_cb = QCheckBox("Gaussian Noise")
        self.gauss_cb.setChecked(self.gaussian_noise)
        self.gauss_cb.stateChanged.connect(lambda: self.set_flag('gaussian_noise', self.gauss_cb.isChecked()))
        self.controlArea.layout().addWidget(self.gauss_cb)


        self.run_button = QPushButton("Generate Augmented Images")
        self.run_button.clicked.connect(self.generate_augmentations)
        self.controlArea.layout().addWidget(self.run_button)
        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def layout_mainArea(self):
        # Font for titles
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)

        # ---- Before Group ----
        before_group = QGroupBox("Before")
        before_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid gray; margin-top: 10px; }")
        before_layout = QVBoxLayout()

        self.image_label = QLabel("No preview available")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(256, 256)
        before_layout.addWidget(self.image_label)

        before_group.setLayout(before_layout)

        # ---- After Group ----
        after_group = QGroupBox("After")
        after_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid gray; margin-top: 10px; }")
        after_layout = QVBoxLayout()

        self.aug_image_label = QLabel("No preview available")
        self.aug_image_label.setAlignment(Qt.AlignCenter)
        self.aug_image_label.setMinimumSize(256, 256)
        after_layout.addWidget(self.aug_image_label)

        after_group.setLayout(after_layout)

        # ---- Add to mainArea ----
        layout = self.mainArea.layout()
        layout.addWidget(before_group)
        layout.addWidget(after_group)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Augmented Images")
        if folder:
            self.save_folder = folder
            self.folder_label.setText(folder)

    def set_augment_count(self, val):
        self.augment_count = val

    def set_flag(self, attr, val):
        setattr(self, attr, val)
        self.show_preview()

    @Inputs.images
    def set_data(self, table):
        self.image_table = table
        self.show_preview()

    def apply_custom_transforms(self, img: Image.Image) -> Image.Image:
        if self.blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
        if self.gaussian_noise:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 15, arr.shape).astype(np.float32)
            arr += noise
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        return img


    def generate_augmentations(self):
        if not self.image_table or not self.save_folder:
            self.error("No image data or folder selected.")
            return

        self.progressBarInit()
        metas = []
        image_col = None
        origin = None

        domain = self.image_table.domain

        for var in domain.metas:
            if var.attributes.get("type") == "image":
                image_col = var
                origin = var.attributes.get("origin")
                var.attributes["origin"] = self.save_folder
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

        new_rows = []
        new_y = []
        
        for i, row in enumerate(self.image_table):
            path = row.metas[self.image_table.domain.metas.index(image_col)]
            try:
                img = load_img(os.path.join(origin,path))
            except Exception:
                print("Could not load "+os.path.join(origin,path))
                continue

            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            gen = datagen.flow(x, batch_size=1)

            for j in range(self.augment_count):
                batch = next(gen)
                aug_img = array_to_img(batch[0])
                aug_img = self.apply_custom_transforms(aug_img)
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

    def show_preview(self):
        if not self.image_table:
            self.image_label.setText("No data")
            return

        domain = self.image_table.domain
        image_col = None
        origin = None

        for var in domain.metas:
            if var.attributes.get("type") == "image":
                image_col = var
                origin = var.attributes.get("origin")
                break

        if not image_col or not origin:
            self.image_label.setText("No image found")
            return

        try:
            # Get the first image path
            path = self.image_table[0].metas[domain.metas.index(image_col)]
            full_path = os.path.join(origin, path)
            image = Image.open(full_path)
            image.thumbnail((256, 256))

            aug_image = self.augment_image(image)
            image.save("/tmp/__preview.png")  # Temporary path to convert to pixmap
            aug_image.save('/tmp/__aug_preview.png')
            pixmap = QPixmap("/tmp/__preview.png")
            aug_pixmap = QPixmap("/tmp/__aug_preview.png")
            self.image_label.setPixmap(pixmap)
            self.aug_image_label.setPixmap(aug_pixmap)
        except Exception as e:
            self.image_label.setText(f"Error loading preview:\n{str(e)}")

    def augment_image(self, image: Image) -> Image:
        datagen = ImageDataGenerator(
            zoom_range=0.2 if self.zoom else 0.0,
            horizontal_flip=self.flip,
            rotation_range=30 if self.rotate else 0,
            shear_range=0.2 if self.shear else 0.0,
            brightness_range=(0.7, 1.3) if self.brightness else None,
            fill_mode='nearest')
        x = img_to_array(image)
        x = x.reshape((1,) + x.shape)
        gen = datagen.flow(x, batch_size=1)
        batch = next(gen)
        aug_img = array_to_img(batch[0])
        aug_img = self.apply_custom_transforms(aug_img)
        return aug_img

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.imageanalytics.import_images import ImportImages, scan
    import_images = ImportImages()
    data_dir = "/home/chris/Downloads/BilddatenLungenentzuendung/training/krank"
    images = import_images.image_meta(scan(data_dir))
    data, err = import_images(data_dir)
    WidgetPreview(OWImageAugmenter).run(data)