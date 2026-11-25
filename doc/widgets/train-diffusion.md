# Train Diffusion

Train a miniature diffusion model: the idea behind Stable Diffusion without the latent space or text encoder.

**Inputs**

- Images: an image table. If it has a class, the model can be conditioned on it.

**Outputs**

- Diffusion Model: for [Generate Images](generate-images.md)

**How it works**

Each training image is mixed with a random amount of noise, and a small U-Net learns to predict the noise that was added. To generate, the model starts from pure noise and removes a little predicted noise at each of the *noise steps*. With **Condition on the class**, the model is also told each image's class (a stand-in for a text prompt); the class is hidden 10% of the time so that *guidance* can later push images toward a class.

**Settings**

- **Height, Width, Channels**: images are resized to this. Height and width must be divisible by 4.
- **Epochs**, **Batch size**, **Learning rate**, **Filters**.
- **Noise steps**: more steps give finer denoising and slower generation.
- **Preview every**: epochs between preview drawings (each takes one pass per noise step).
