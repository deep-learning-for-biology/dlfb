import jax
import jax.numpy as jnp
from transformers import AutoImageProcessor

IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained("microsoft/resnet-50")


def skew(
  image: jax.Array, size: tuple[int, int, int] = (224, 224, 3)
) -> jax.Array:
  """Rescales and resizes image to fixed size using bilinear interpolation."""
  image = rescale_image(image)
  image = jax.image.resize(image, size, method="bilinear")
  return image


def crop(image: jax.Array) -> jax.Array:
  """Rescales, resizes with preserved aspect ratio, then center-crops image."""
  image = rescale_image(image)
  image = resize_preserve_aspect(image, 256)
  image = center_crop(image, 224)
  return image


def resize_preserve_aspect(
  image: jax.Array, short_side: int = 256
) -> jax.Array:
  """Resize image with shorter side is `short_side`, keeping aspect ratio."""
  h, w, c = image.shape
  scale = short_side / jnp.minimum(h, w)
  new_h = jnp.round(h * scale).astype(jnp.int32)
  new_w = jnp.round(w * scale).astype(jnp.int32)
  resized = jax.image.resize(image, (new_h, new_w, c), method="bilinear")
  return resized


def center_crop(image: jax.Array, size: int = 224) -> jax.Array:
  """Crop the center square of given size from an image."""
  h, w, _ = image.shape
  top = (h - size) // 2
  left = (w - size) // 2
  return image[top : top + size, left : left + size]


def rescale_image(image: jax.Array) -> jax.Array:
  """Normalizes pixel values to the [0, 1] range by dividing by 255."""
  return image / 255.0


def resnet(image: jax.Array) -> jax.Array:
  """Preprocess from pretrained model with transpose for compatibility."""
  image = IMAGE_PROCESSOR(image, return_tensors="jax", do_rescale=True)
  image = image["pixel_values"]
  image = convert_nchw_to_nhwc(image)
  return image


def convert_nchw_to_nhwc(image: jax.Array) -> jax.Array:
  """Convert image from NCHW to NHWC format."""
  return jnp.transpose(image, (0, 2, 3, 1))


def skew_resnet(image: jax.Array) -> jax.Array:
  """Applies skew resizing followed by ResNet normalization."""
  image = skew(image)
  image = resnet_normalize_image(image)
  return image


def crop_resnet(image: jax.Array) -> jax.Array:
  """Applies center crop followed by ResNet normalization."""
  image = crop(image)
  image = resnet_normalize_image(image)
  return image


def resnet_normalize_image(image: jax.Array) -> jax.Array:
  """Applies ResNet-style normalization using fixed mean and std."""
  mean = jnp.array([0.485, 0.456, 0.406])
  std = jnp.array([0.229, 0.224, 0.225])
  return (image - mean) / std
