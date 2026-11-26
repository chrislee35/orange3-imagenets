# Misclassified Images

A confusion matrix of a classification, with the wrongly classified images as output.

**Inputs**

- Annotated Data: the output of [Classify Images](classify-images.md) (needs a class and a `Prediction` column)

**Outputs**

- Misclassified: every image whose prediction differs from its class
- Selected: the images in the confusion-matrix cell you click (true class in the row, prediction in the column)

Rows are the true classes, columns the predictions, and the **recall** column gives the share of each class that was recognized. Green cells on the diagonal are correct; red cells are mistakes, the darker the more frequent. Connect either output to *Image Viewer* to see the pictures.
