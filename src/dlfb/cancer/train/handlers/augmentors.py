import dm_pix as pix
import jax


@jax.jit
def flipping_augmentor(image: jax.Array, rng: jax.Array) -> jax.Array:
  """Applies random horizontal and vertical flips."""
  image = pix.random_flip_left_right(rng, image)
  image = pix.random_flip_up_down(rng, image)
  return image


@jax.jit
def rich_augmentor(image: jax.Array, rng: jax.Array) -> jax.Array:
  """Applies random flips, brightness, contrast, hue changes, and rotation."""
  image = pix.random_flip_left_right(rng, image)
  image = pix.random_flip_up_down(rng, image)
  image = pix.random_brightness(rng, image, max_delta=0.1)
  image = pix.random_contrast(rng, image, lower=0.9, upper=1)
  image = pix.random_hue(rng, image, max_delta=0.05)
  # Angles are provided in radians, i.e. +/- 10 degrees.
  image = pix.rotate(
    image,
    angle=jax.random.uniform(rng, shape=(), minval=-0.174533, maxval=0.174533),
  )
  return image
