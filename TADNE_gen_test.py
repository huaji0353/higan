import numpy as np
import cv2
import PIL.Image
import torch

def imshow(images, col, viz_size=256):
  """Shows images in one figure."""
  num, height, width, channels = images.shape
  assert num % col == 0
  row = num // col
  fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)
  for idx, image in enumerate(images):
    i, j = divmod(idx, col)
    y = i * viz_size
    x = j * viz_size
    if height != viz_size or width != viz_size:
      image = cv2.resize(image, (viz_size, viz_size))
    fused_image[y:y + viz_size, x:x + viz_size] = image
  fused_image = np.asarray(fused_image, dtype=np.uint8)
  PIL.Image.fromarray(fused_image).save('imshow.jpg')

def postprocess(images):
  """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
  images = images.detach().cpu().numpy()
  images = (images + 1) * 255 / 2
  images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
  images = images.transpose(0, 2, 3, 1)
  return images
  
  
from models.stylegan2_generator import StyleGAN2Generator

checkpoint = torch.load('/content/stylegan2-genforce/models/pretrain/pytorch/stylegan2_TADNE.pth', map_location='cuda')
checkpoint['truncation.truncation'] = torch.randn(1, 16, 1)

generator = TADNEGeneratorNet().to('cuda')
#generator = StyleGAN2Generator('stylegan2_TADNE').to('cuda')

generator.load_state_dict(checkpoint)
generator.eval()

# np.random.seed(random_seed)
# codes = np.random.randn(1, 1024)
# generator.mapping(torch.from_numpy(codes).type(torch.FloatTensor).to('cuda'))

num_samples = 1
random_seed = 35363
np.random.seed(random_seed)

with torch.no_grad():
  zcodes = torch.from_numpy(np.random.randn(num_samples, generator.z_space_dim)).type(torch.FloatTensor).to('cuda')
  images = generator(zcodes)
imshow(postprocess(images), col=num_samples)
