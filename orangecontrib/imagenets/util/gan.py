"""A small DCGAN: a generator that turns random noise into images and a discriminator that tries to tell them from real ones.

Qt-free so it can be tested and reused; the Train GAN widget only adds the controls and the live plots.
"""

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from orangecontrib.imagenets.util.image_loader import to_signed, to_uint8


class GANModel:
    """A trained generator plus what is needed to use it."""

    kind = "gan"

    def __init__(self, generator, latent_dim, shape):
        self.generator = generator
        self.latent_dim = latent_dim
        self.shape = tuple(shape)  # (height, width, channels)

    def sample(self, n, seed=None):
        """Return ``n`` generated images as uint8 (n, height, width, channels)."""
        z = np.random.default_rng(seed).standard_normal((n, self.latent_dim)).astype(np.float32)
        return to_uint8(self.generator.predict(z, verbose=0))


def check_shape(shape):
    h, w, _ = shape
    if h % 4 or w % 4:
        raise ValueError(f"Image height and width must be divisible by 4 (got {w}x{h}).")


def build_generator(shape, latent_dim=64, base_filters=32):
    check_shape(shape)
    h, w, c = shape
    return keras.Sequential(
        [
            keras.Input((latent_dim,)),
            layers.Dense((h // 4) * (w // 4) * base_filters * 2),
            layers.LeakyReLU(0.2),
            layers.Reshape((h // 4, w // 4, base_filters * 2)),
            layers.Conv2DTranspose(base_filters, 4, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(c, 4, strides=2, padding="same", activation="tanh"),
        ],
        name="generator",
    )


def build_discriminator(shape, base_filters=32):
    check_shape(shape)
    return keras.Sequential(
        [
            keras.Input(shape),
            layers.Conv2D(base_filters, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Conv2D(base_filters * 2, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )


def train_gan(images, epochs=10, batch_size=64, latent_dim=64, base_filters=32, learning_rate=2e-4, seed=0,
              on_epoch=None, should_stop=None):
    """Train a GAN on ``images`` (float, n x h x w x c, 0..255).

    ``on_epoch(epoch, d_loss, g_loss, d_acc, samples)`` is called after every epoch with 16 images (uint8) generated from
    the same noise each time, so a caller can show the generator improving. ``should_stop()`` returning True ends training
    early. Returns a ``GANModel``.

    Each batch takes two steps: the discriminator learns to separate real from generated images, then the generator learns
    to fool the (unchanged) discriminator. They are written as two explicit gradient steps rather than a frozen, compiled
    "combined model", because Keras 3 does not reliably honor ``trainable = False`` after compile.
    """
    keras.utils.set_random_seed(seed)
    shape = images.shape[1:]
    x = to_signed(images)
    generator = build_generator(shape, latent_dim, base_filters)
    discriminator = build_discriminator(shape, base_filters)
    g_opt = keras.optimizers.Adam(learning_rate, beta_1=0.5)
    d_opt = keras.optimizers.Adam(learning_rate, beta_1=0.5)
    bce = keras.losses.BinaryCrossentropy()

    @tf.function
    def d_step(real, noise):
        with tf.GradientTape() as tape:
            fake = generator(noise, training=True)
            p_real = discriminator(real, training=True)
            p_fake = discriminator(fake, training=True)
            loss = bce(tf.fill(tf.shape(p_real), 0.9), p_real) + bce(tf.zeros_like(p_fake), p_fake)  # 0.9: one-sided smoothing
        grads = tape.gradient(loss, discriminator.trainable_variables)
        d_opt.apply(grads, discriminator.trainable_variables)
        acc = (tf.reduce_mean(tf.cast(p_real > 0.5, tf.float32)) + tf.reduce_mean(tf.cast(p_fake < 0.5, tf.float32))) / 2
        return loss / 2, acc

    @tf.function
    def g_step(noise):
        with tf.GradientTape() as tape:
            p_fake = discriminator(generator(noise, training=True), training=True)
            loss = bce(tf.ones_like(p_fake), p_fake)  # the generator wants its images called "real"
        grads = tape.gradient(loss, generator.trainable_variables)
        g_opt.apply(grads, generator.trainable_variables)
        return loss

    rng = np.random.default_rng(seed)
    preview_noise = rng.standard_normal((16, latent_dim)).astype(np.float32)
    n = len(x)
    steps = max(1, n // batch_size)
    batch_size = min(batch_size, n)

    for epoch in range(1, epochs + 1):
        d_losses, g_losses, d_accs = [], [], []
        order = rng.permutation(n)
        for step in range(steps):
            real = x[order[step * batch_size: (step + 1) * batch_size]]
            noise = rng.standard_normal((len(real), latent_dim)).astype(np.float32)
            d_loss, d_acc = d_step(real, noise)
            g_loss = g_step(rng.standard_normal((len(real), latent_dim)).astype(np.float32))
            d_losses.append(float(d_loss))
            d_accs.append(float(d_acc))
            g_losses.append(float(g_loss))
        if on_epoch:
            samples = to_uint8(generator.predict(preview_noise, verbose=0))
            on_epoch(epoch, float(np.mean(d_losses)), float(np.mean(g_losses)), float(np.mean(d_accs)), samples)
        if should_stop and should_stop():
            break
    return GANModel(generator, latent_dim, shape)
