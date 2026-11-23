# Save ImageNet

Save a model to disk.

![Saving the ImageNet model](../images/save-model.png)

**Inputs**

- Model: a Keras model

**Using it**

- **Save as .h5** stores the whole model, weights included.
- **Save as JSON** stores only the architecture. Keep the weights in a file named `<name>_weights.h5` next to it so [Load ImageNet](load-imagenet.md) can find them.
- The `.h5` or `.json` extension is added if you leave it off.
