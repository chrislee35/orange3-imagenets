import os
import numpy as np
import cv2

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QPixmap, QFont
from AnyQt.QtWidgets import QLabel, QVBoxLayout, QFileDialog, QPushButton, QCheckBox, QSpinBox, QHBoxLayout, QFrame, QGroupBox
from PyQt5.QtCore import QThread, pyqtSignal

from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, StringVariable

from orangecontrib.imagenets.util.image_table import image_table_variables

class PreprocessWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, data, output_dir, do_grayscale, do_resize, resize_width, resize_height, do_normalize):
        super().__init__()
        self.data = data
        self.output_dir = output_dir
        self.do_grayscale = do_grayscale
        self.do_resize = do_resize
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.do_normalize = do_normalize
        self.origin, self.image_col_index = image_table_variables(data)

    def run(self):
        total = len(self.data)

        for i, row in enumerate(self.data):
            img_path = os.path.join(self.origin, row.metas[self.image_col_index])
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if self.do_grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.do_resize:
                img = cv2.resize(img, (self.resize_width, self.resize_height))
            if self.do_normalize:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
            out_path = os.path.join(self.output_dir, row.metas[self.image_col_index])
            out_dir = os.path.dirname(out_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(out_path, img)
            self.progress.emit(int(100 * (i + 1) / total))

        data = self.data.copy()
        data.domain.metas[self.image_col_index].attributes["origin"] = self.output_dir
        self.finished.emit(data)

class OWImagePreprocessor(OWWidget):
    name = "Preprocess Images"
    description = "Resize and normalize images, saving to disk."
    icon = "icons/preprocess.svg"
    priority = 10

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        preprocessed_data = Output("Preprocessed Data", Table)

    do_grayscale = Setting(False)
    do_resize = Setting(True)
    resize_width = Setting(224)
    resize_height = Setting(224)
    do_normalize = Setting(False)

    def __init__(self):
        super().__init__()
        self.data = None
        self.output_dir = None
        self.layout_controlArea()
        self.layout_mainArea()

    def layout_controlArea(self):
        select_button = QPushButton("Select Output Folder")
        select_button.clicked.connect(self.select_folder)
        self.info_label = QLabel("Waiting for input...")

        self.gray_cb = QCheckBox("Convert to Grayscale")
        self.gray_cb.setChecked(self.do_grayscale)
        self.gray_cb.stateChanged.connect(lambda: self.settings_changed("do_grayscale", self.gray_cb.isChecked()))

        self.resize_cb = QCheckBox("Resize Images")
        self.resize_cb.setChecked(self.do_resize)
        self.resize_cb.stateChanged.connect(lambda: self.settings_changed("do_resize", self.resize_cb.isChecked()))

        self.width_spin = QSpinBox()
        self.width_spin.setRange(32, 2048)
        self.width_spin.setValue(self.resize_width)
        self.width_spin.valueChanged.connect(lambda val: self.settings_changed("resize_width", val))

        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 2048)
        self.height_spin.setValue(self.resize_height)
        self.height_spin.valueChanged.connect(lambda val: self.settings_changed("resize_height", val))

        self.norm_cb = QCheckBox("Normalize Images")
        self.norm_cb.setChecked(self.do_normalize)
        self.norm_cb.stateChanged.connect(lambda: self.settings_changed("do_normalize", self.norm_cb.isChecked()))

        size_frame = QFrame()
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Width:"))
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("Height:"))
        size_layout.addWidget(self.height_spin)
        size_frame.setLayout(size_layout)

        a = self.controlArea.layout().addWidget
        a(self.info_label)
        a(select_button)
        a(self.gray_cb)
        a(self.norm_cb)
        a(self.resize_cb)
        a(size_frame)

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

    def settings_changed(self, attr, value):
        setattr(self, attr, value)
        self.show_preview_image()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.output_dir = folder
            self.try_preprocess()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        #self.try_preprocess()
        self.show_preview_image()

    def try_preprocess(self):
        if self.data is not None and self.output_dir:
            self.info_label.setText("Preprocessing...")
            self.progressBarInit()

            self.worker = PreprocessWorker(
                self.data, self.output_dir,
                self.do_grayscale,
                self.do_resize,
                self.resize_width,
                self.resize_height,
                self.do_normalize
            )
            self.worker.progress.connect(self.progressBarSet)
            self.worker.finished.connect(self.handle_preprocessed)
            self.worker.start()

    def handle_preprocessed(self, table: Table):
        self.Outputs.preprocessed_data.send(table)
        self.progressBarFinished()
        self.info_label.setText("Preprocessing complete.")

    def preprocess_image(self, img_path: str) -> QPixmap:
        img = cv2.imread(img_path)
        if self.do_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.do_resize:
            img = cv2.resize(img, (self.resize_width, self.resize_height))
        if self.do_normalize:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
        is_success, buffer = cv2.imencode(".png", img)
        pixmap = QPixmap()
        pixmap.loadFromData(buffer, "png")
        return pixmap

    def show_preview_image(self):
        if not self.data:
            self.info_label.setText("No data")
            return

        origin, image_col_index = image_table_variables(self.data)

        try:
            # Get the first image path
            path = self.data[0].metas[image_col_index]
            full_path = os.path.join(origin, path)
            pixmap = QPixmap(full_path)
            self.image_label.setPixmap(pixmap)
            preprocessed_pixmap = self.preprocess_image(full_path)
            self.aug_image_label.setPixmap(preprocessed_pixmap)
        except Exception as e:
            self.image_label.setText(f"Error loading preview:\n{str(e)}")

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.imageanalytics.import_images import ImportImages, scan
    import_images = ImportImages()
    data_dir = "/media/veracrypt1/work/competition/animals"
    images = import_images.image_meta(scan(data_dir))
    data, err = import_images(data_dir)
    WidgetPreview(OWImagePreprocessor).run(data)