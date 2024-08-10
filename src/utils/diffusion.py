from __future__ import annotations

import asyncio
import base64
import gc
import logging
import time
import tomllib
from io import BytesIO
from typing import TYPE_CHECKING

import torch
from diffusers import DiffusionPipeline

if TYPE_CHECKING:
  from PIL import Image

with open("config.toml") as f:
  config = tomllib.loads(f.read())

LOG = logging.getLogger(__name__)
MODEL_ID = config["ai"]["model"]
DEVICE = config["ai"]["device"]
MODEL_LOCK = asyncio.Lock()
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
pipe.to(DEVICE)

if not torch.cuda.is_available() and "cuda" in DEVICE:
  raise Exception("Cuda is not available for model inference!")


def generate_images(
  prompt: str,
  *,
  negative_prompt: str = None,
  width: int = None,
  height: int = None,
) -> list[Image.Image]:
  output = pipe(
    prompt, negative_prompt=negative_prompt, width=width, height=height
  )
  images: list[Image.Image] = output.images
  return images


def images_to_base64(images: list[Image.Image]) -> list[str]:
  output = []

  for image in images:
    b = BytesIO()
    image.save(b, format="png")
    b.seek(0)
    output.append(base64.b64encode(b.read()).decode())

  return output


async def generate(
  prompt: str,
  *,
  negative_prompt: str = None,
  width: int = None,
  height: int = None,
) -> list[str]:
  "Take in prompt and some parameters, return list of base64-encoded PNGs."
  loop = asyncio.get_running_loop()
  start = time.time()

  def t():
    return round(time.time() - start, 4)

  if width is not None and width % 8 != 0:
    raise ValueError("Width must be a multiple of 8!")
  if height is not None and height % 8 != 0:
    raise ValueError("Height must be a multiple of 8!")

  LOG.info(f"[{t()}] Waiting for lock...")
  async with MODEL_LOCK:
    LOG.info(f"[{t()}] Lock acquired. Generating images...")
    images = await loop.run_in_executor(
      None,
      lambda: generate_images(
        prompt, negative_prompt=negative_prompt, width=width, height=height
      ),
    )
    LOG.info(f"[{t()}] Images generated, transcribing to base64...")
    b64 = await loop.run_in_executor(None, images_to_base64, images)
    LOG.info(f"[{t()}] Base64 acquired. Finishing up...")
  cleanup()
  LOG.info(f"[{t()}] All cleaned up!")
  return b64


def cleanup():
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()
  gc.collect()
