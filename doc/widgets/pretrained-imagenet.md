# Pretrained ImageNet

A network that already knows how to see, ready to learn your classes (transfer learning).

**Inputs**

- Images (optional): an image table. Its class values set the number of **Classes**.

**Outputs**

- Model: a Keras model for [Train ImageNet](train-imagenet.md) and [Classify Images](classify-images.md).

The network is a small MobileNetV2 trained on 1.2 million photos (the ImageNet collection), with a new last layer for your classes. Images are loaded in color at the network's size (96 x 96 or 128 x 128) and scaled for it; you do not add a *Rescaling* layer.

- **Pretrained network**: *small* (96 x 96, 410,208 weights before the new layer) or *medium* (128 x 128).
- **Classes**: the number of classes in your data.
- **Freeze the pretrained part**: when ticked (the default) only the new last layer is trained. That is fast and needs little data. Untick it to fine-tune every weight, which is slower and can overfit small data.

The first use downloads the weights (2 to 4 MB) from Google's Keras servers and keeps them for later. On 5,000 CIFAR-10 images, five epochs with the frozen network take about 35 seconds on a CPU and reach about 77% to 79% test accuracy, against about 50% for [SimpleCNN](build-imagenet.md) trained from scratch.
