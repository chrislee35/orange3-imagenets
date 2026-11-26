# Train ImageNet

Train a Keras model on an image table and watch its performance.

![Training graph](../images/training-graph.png)

**Inputs**

- Learner: the model from [Build ImageNet](build-imagenet.md)
- Evaluation Data: an image table with a class variable

**Outputs**

- Trained Model: the model after training

**Settings**

- **Batch size** (default 32) and **Epochs** (default 10). For dropout, add a *Dropout* layer in *Build ImageNet*.
