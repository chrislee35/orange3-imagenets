"""A miniature denoising diffusion model (DDPM), the idea behind Stable Diffusion without the latent space or text encoder.

Training: take a real image, mix in a random amount of noise (the timestep t says how much), and teach a small U-Net to
predict the noise that was added. Generating: start from pure noise and repeatedly subtract the predicted noise.
Optionally the net is told the image's class (a stand-in for a text prompt); randomly hiding the class while training lets
the sampler push images toward a class with "classifier-free guidance".

Qt-free so it can be tested and reused; the Train Diffusion widget only adds the controls and the live plots.
"""

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from orangecontrib.imagenets.util.image_loader import to_signed, to_uint8

TIME_DIM = 64


def make_schedule(steps):
    """Linear noise schedule, rescaled so the image is nearly pure noise at the last step whatever ``steps`` is."""
    scale = 1000 / steps
    betas = np.linspace(1e-4 * scale, 0.02 * scale, steps, dtype=np.float64)
    alphas = 1.0 - betas
    return betas, alphas, np.cumprod(alphas)


def time_embedding(t, dim=TIME_DIM):
    """Sinusoidal embedding of integer timesteps (n,) -> (n, dim)."""
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    args = np.asarray(t, np.float64)[:, None] * freqs[None]
    return np.concatenate([np.sin(args), np.cos(args)], axis=1).astype(np.float32)


def build_denoiser(shape, num_classes=0, base_filters=32):
    """U-Net that maps (noisy image, timestep embedding[, class]) to the predicted noise."""
    h, w, c = shape
    if h % 4 or w % 4:
        raise ValueError(f"Image height and width must be divisible by 4 (got {w}x{h}).")
    img = keras.Input(shape, name="noisy_image")
    t_in = keras.Input((TIME_DIM,), name="time")
    inputs = [img, t_in]
    emb = layers.Dense(128, activation="swish")(t_in)
    if num_classes:
        label = keras.Input((1,), dtype="int32", name="label")  # value num_classes means "no class given"
        inputs.append(label)
        emb = emb + layers.Flatten()(layers.Embedding(num_classes + 1, 128)(label))
    emb = layers.Dense(128, activation="swish")(emb)

    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="swish")(x)
        bias = layers.Reshape((1, 1, filters))(layers.Dense(filters)(emb))  # how noisy / which class, added to every pixel
        x = layers.Conv2D(filters, 3, padding="same", activation="swish")(x + bias)
        return x

    d1 = block(img, base_filters)
    d2 = block(layers.AveragePooling2D(2)(d1), base_filters * 2)
    mid = block(layers.AveragePooling2D(2)(d2), base_filters * 4)
    u2 = block(layers.Concatenate()([layers.UpSampling2D(2)(mid), d2]), base_filters * 2)
    u1 = block(layers.Concatenate()([layers.UpSampling2D(2)(u2), d1]), base_filters)
    return keras.Model(inputs, layers.Conv2D(c, 1)(u1), name="denoiser")


class DiffusionModel:
    """A trained denoiser plus the noise schedule needed to generate with it."""

    kind = "diffusion"

    def __init__(self, net, shape, steps, class_names=None):
        self.net = net
        self.shape = tuple(shape)
        self.steps = steps
        self.class_names = list(class_names) if class_names else None
        self.betas, self.alphas, self.alpha_bars = make_schedule(steps)

    def _predict_noise(self, x, t, labels, guidance):
        temb = np.repeat(time_embedding([t]), len(x), axis=0)
        if self.class_names is None:
            return self.net([x, temb], training=False).numpy()
        n_cls = len(self.class_names)
        cond = self.net([x, temb, labels.reshape(-1, 1)], training=False).numpy()
        if guidance == 0 or (labels == n_cls).all():
            return cond
        uncond = self.net([x, temb, np.full((len(x), 1), n_cls, np.int32)], training=False).numpy()
        return (1 + guidance) * cond - guidance * uncond

    def sample(self, n, labels=None, guidance=2.0, seed=None, keep_steps=0, on_step=None):
        """Generate ``n`` images as uint8 (n, h, w, c) by denoising pure noise.

        ``labels``: class index per image (list, or one int for all); None means unconditioned. ``guidance`` > 0 pushes
        images toward their class. If ``keep_steps`` > 0, also returns that many evenly spaced intermediate snapshots, a list
        of uint8 arrays from noisiest to final. ``on_step(done, total)`` reports progress.
        """
        rng = np.random.default_rng(seed)
        if self.class_names is not None:
            n_cls = len(self.class_names)
            labels = np.full(n, n_cls, np.int32) if labels is None else np.broadcast_to(np.asarray(labels, np.int32), (n,)).copy()
        x = rng.standard_normal((n, *self.shape)).astype(np.float32)
        snap_at = set(np.linspace(self.steps - 1, 0, keep_steps).round().astype(int)) if keep_steps else set()
        snaps = []
        for done, t in enumerate(range(self.steps - 1, -1, -1), start=1):
            eps = self._predict_noise(x, t, labels, guidance)
            a, ab, beta = self.alphas[t], self.alpha_bars[t], self.betas[t]
            x = (x - (1 - a) / np.sqrt(1 - ab) * eps) / np.sqrt(a)
            if t > 0:
                x = x + np.sqrt(beta) * rng.standard_normal(x.shape)
            x = x.astype(np.float32)
            if t in snap_at:
                snaps.append(to_uint8(np.clip(x, -1, 1)))
            if on_step:
                on_step(done, self.steps)
        out = to_uint8(np.clip(x, -1, 1))
        return (out, snaps) if keep_steps else out


def train_diffusion(images, labels=None, epochs=10, batch_size=64, steps=200, base_filters=16, learning_rate=1e-3,
                    seed=0, on_epoch=None, should_stop=None):
    """Train a diffusion model on ``images`` (float, n x h x w x c, 0..255).

    ``labels`` (optional): class name per image, to make a class-conditioned model.
    ``on_epoch(epoch, loss, model)`` is called after every epoch, so a caller can plot the loss or sample a few images
    (10% of the time the class is hidden, which is what makes guidance possible). ``should_stop()`` returning True ends
    training early. Returns a ``DiffusionModel``.
    """
    keras.utils.set_random_seed(seed)
    shape = images.shape[1:]
    x0 = to_signed(images)
    class_names = sorted(set(labels)) if labels is not None else None
    y = np.array([class_names.index(v) for v in labels], np.int32) if labels is not None else None
    net = build_denoiser(shape, len(class_names) if class_names else 0, base_filters)
    model = DiffusionModel(net, shape, steps, class_names)
    sqrt_ab = np.sqrt(model.alpha_bars).astype(np.float32)
    sqrt_1m_ab = np.sqrt(1 - model.alpha_bars).astype(np.float32)
    opt = keras.optimizers.Adam(learning_rate)

    @tf.function
    def step(xt, temb, label, eps):
        with tf.GradientTape() as tape:
            inputs = [xt, temb] if label is None else [xt, temb, label]
            loss = tf.reduce_mean(tf.square(eps - net(inputs, training=True)))
        grads = tape.gradient(loss, net.trainable_variables)
        opt.apply(grads, net.trainable_variables)
        return loss

    rng = np.random.default_rng(seed)
    n = len(x0)
    batch_size = min(batch_size, n)
    n_steps = max(1, n // batch_size)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        losses = []
        for i in range(n_steps):
            idx = order[i * batch_size: (i + 1) * batch_size]
            t = rng.integers(0, steps, len(idx))
            eps = rng.standard_normal((len(idx), *shape)).astype(np.float32)
            xt = sqrt_ab[t][:, None, None, None] * x0[idx] + sqrt_1m_ab[t][:, None, None, None] * eps
            label = None
            if y is not None:
                label = y[idx].copy().reshape(-1, 1)
                label[rng.random(len(idx)) < 0.1] = len(class_names)
            losses.append(float(step(xt, time_embedding(t), label, eps)))
        if on_epoch:
            on_epoch(epoch, float(np.mean(losses)), model)
        if should_stop and should_stop():
            break
    return model
