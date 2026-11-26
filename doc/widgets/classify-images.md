# Classify Images

Predict the class of every image in a table.

![Viewing the classification results](../images/classified-data-table.png)

**Inputs**

- Data: an image table
- Model: a trained Keras model

**Outputs**

- Annotated Data: the input with a `Prediction` meta column

Images are loaded at the size and color depth the model was built for. Classification starts as soon as both inputs are connected. If the data has a class, the widget shows the accuracy; connect the output to [Misclassified Images](misclassified-images.md) to see the mistakes.
