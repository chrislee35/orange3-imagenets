# Generate Images

Draw new images with a trained GAN generator or diffusion model.

**Inputs**

- Model: from [Train GAN](train-gan.md) or [Train Diffusion](train-diffusion.md)

**Outputs**

- Images: an image table (PNG files in a temporary folder), usable by Image Viewer, Train ImageNet and Classify Images. It has a class when you chose one.

**Settings**

- **Images**, **Random seed** (same seed, same images).
- **Class** (class-conditioned diffusion models): which class to draw.
- **Guidance** (diffusion): how strongly to push images toward the class. 0 ignores the class.
- **Show denoising steps** (diffusion): shows the first images at six points from noise to finished picture, finished images in the last row.
