# Train GAN

Train a small generative adversarial network (GAN) on an image table.

**Inputs**

- Images: an image table (class labels are ignored)

**Outputs**

- Generator: the trained generator, for [Generate Images](generate-images.md)

**How it works**

A *generator* turns a random noise vector into an image. A *discriminator* is shown real and generated images and learns to tell them apart. Each batch, the discriminator improves, then the generator improves at fooling it. The plot shows both losses and the discriminator's accuracy; when it hovers near 0.5 the two are evenly matched. After every epoch the widget draws 16 images from the same noise, so you can watch the generator improve.

**Settings**

- **Height, Width, Channels**: images are resized to this. Height and width must be divisible by 4.
- **Epochs**, **Batch size**, **Learning rate**.
- **Noise size**: length of the generator's random input. **Filters**: width of both networks.
- **Stop** ends training early and keeps the generator so far.
