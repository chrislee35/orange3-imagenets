# Train ImageNet

Train a Keras model on an image table and watch its performance.

![Training graph](../images/training-graph.png)

**Inputs**

- Learner: the model from [Build ImageNet](build-imagenet.md)
- Evaluation Data: an image table with a class variable

**Outputs**

- Trained Model: the model after training

**Settings**

- **Batch size** (default 32), **Epochs** (default 10) and **Learning rate** (default 0.001; use 0.0001 or lower to fine-tune a pretrained network). For dropout, add a *Dropout* layer in *Build ImageNet*.
