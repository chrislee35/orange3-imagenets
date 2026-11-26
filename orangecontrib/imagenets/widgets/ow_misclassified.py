import numpy as np
from AnyQt.QtGui import QBrush, QColor
from AnyQt.QtWidgets import QAbstractItemView, QLabel, QTableWidget, QTableWidgetItem
from Orange.data import Table
from Orange.widgets import gui, widget
from Orange.widgets.widget import Input, Output

from orangecontrib.imagenets.util.confusion import confusion_matrix, true_and_predicted


class OWMisclassified(widget.OWWidget):
    name = "Misclassified Images"
    description = "Confusion matrix of a classification, with the wrongly classified images as output."
    icon = "icons/misclassified.svg"
    priority = 25
    keywords = "confusion, errors, mistakes, wrong"

    class Inputs:
        data = Input("Annotated Data", Table)

    class Outputs:
        misclassified = Output("Misclassified", Table)
        selected = Output("Selected", Table)

    class Error(widget.OWWidget.Error):
        no_prediction = widget.Msg("Connect the output of Classify Images (the table needs a class and a Prediction column).")

    want_main_area = True

    def __init__(self):
        super().__init__()
        self.resize(1000, 520)
        self.data = None
        self._truth, self._pred = [], []
        self.summary = QLabel("")
        self.controlArea.layout().addWidget(self.summary)
        gui.label(self.controlArea, self, "Rows are the true class, columns the prediction.\n"
                  "Click a cell to send those images to Selected.")
        self.controlArea.layout().addStretch(1)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.cellClicked.connect(self._cell_clicked)
        self.mainArea.layout().addWidget(self.table)
        self._labels = []

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.Error.no_prediction.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.summary.setText("")
        self._labels = []
        self.Outputs.selected.send(None)
        if data is None:
            self.Outputs.misclassified.send(None)
            return
        tp = true_and_predicted(data)
        if tp is None:
            self.Error.no_prediction()
            self.Outputs.misclassified.send(None)
            return
        self._truth, self._pred = tp
        wrong = [i for i, (t, p) in enumerate(zip(*tp)) if t != p]
        self.Outputs.misclassified.send(data[wrong] if wrong else None)
        self._fill()
        self.summary.setText(f"{len(self._truth)} images, {len(wrong)} wrong "
                             f"({1 - len(wrong) / max(len(self._truth), 1):.1%} accuracy)")

    def _fill(self):
        labels, counts = confusion_matrix(self._truth, self._pred)
        self._labels = labels
        n = len(labels)
        self.table.setRowCount(n)
        self.table.setColumnCount(n + 1)
        self.table.setHorizontalHeaderLabels([*labels, "recall"])
        self.table.setVerticalHeaderLabels(labels)
        peak = max(counts.max(initial=0), 1)
        for i in range(n):
            total = counts[i].sum()
            for j in range(n):
                item = QTableWidgetItem(str(counts[i, j]) if counts[i, j] else "")
                shade = int(255 - 160 * counts[i, j] / peak) if counts[i, j] else 255
                color = QColor(shade, 255, shade) if i == j else QColor(255, shade, shade)
                item.setBackground(QBrush(color))
                item.setForeground(QBrush(QColor(0, 0, 0)))
                self.table.setItem(i, j, item)
            recall = QTableWidgetItem(f"{counts[i, i] / total:.0%}" if total else "")
            self.table.setItem(i, n, recall)
        self.table.resizeColumnsToContents()

    def _cell_clicked(self, row, col):
        if col >= len(self._labels):
            return
        t, p = self._labels[row], self._labels[col]
        rows = [i for i, (a, b) in enumerate(zip(self._truth, self._pred)) if a == t and b == p]
        self.Outputs.selected.send(self.data[np.array(rows, int)] if rows else None)
