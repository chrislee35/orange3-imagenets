# Inspect ImageNet

Show the filters a trained network learned and what each layer sees in an image.

**Inputs**

- Model: a trained Keras model (from [Train ImageNet](train-imagenet.md) or [Load ImageNet](load-imagenet.md))
- Images (optional): an image table; choose a **Row** to look at one picture

**What it shows**

- **Filters**: the weights of the chosen convolution layer, one small picture per filter. On a color image each filter is drawn in color. On deeper layers, where a filter has one kernel per incoming map, the kernels are averaged to a gray picture.
- **The image** and its **Feature maps**: what each filter of the chosen layer responds to in that picture. Each map is stretched to full brightness, so bright means a strong response relative to that map's own range.

Only the model's top-level layers are listed (convolutions, pooling, and residual blocks). Layers inside a pretrained base are not.
