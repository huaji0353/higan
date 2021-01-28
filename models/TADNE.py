"""thisanimedoesnotexist

    self.net = TADNEGeneratorNet()
"""

from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stylegan2_generator_network import MappingModule,TruncationModule,SynthesisModule

__all__ = ['TADNEGeneratorNet']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [512]

# Initial resolution.
_INIT_RES = 4

# Architectures allowed.
_ARCHITECTURES_ALLOWED = ['resnet', 'skip', 'origin']

class TADNEGeneratorNet(nn.Module):
  """Defines the generator network in TADNE.

  NOTE: the generated images are with `RGB` color channels and range [-1, 1].
  """

  def __init__(self,
               resolution=512,
               z_space_dim=1024,
               w_space_dim=1024,
               image_channels=3,
               architecture_type='skip',
               fused_modulate=True,
               truncation_psi=0.5,
               truncation_layers=None,
               randomize_noise=False,
               num_mapping_layers=4,
               fmaps_base=32<<11,
               fmaps_max=1024):
    """Initializes the generator with basic settings.

    Args:
      resolution: The resolution of the output image. (default: 1024)
      z_space_dim: The dimension of the initial latent space. (default: 512)
      w_space_dim: The dimension of the disentangled latent vectors, w.
        (default: 512)
      image_channels: Number of channels of output image. (default: 3)
      architecture_type: Defines the architecture type. (default: `resnet`)
      fused_modulate: Whether to fuse `style_modulate` and `conv2d` together.
        (default: True)
      truncation_psi: Style strength multiplier for the truncation trick.
        `None` or `1.0` indicates no truncation. (default: 0.5)
      truncation_layers: Number of layers for which to apply the truncation
        trick. `None` or `0` indicates no truncation. (default: 18)
      randomize_noise: Whether to add random noise for each convolutional layer.
        (default: False)
      num_mapping_layers: Number of fully-connected layers to map Z space to W
        space. (default: 8)
      fmaps_base: Base factor to compute number of feature maps for each layer.
        (default: 32 << 10)
      fmaps_max: Maximum number of feature maps in each layer. (default: 512)

    Raises:
      ValueError: If the input `resolution` is not supported, or
        `architecture_type` is not supported.
    """
    super().__init__()

    if resolution not in _RESOLUTIONS_ALLOWED:
      raise ValueError(f'Invalid resolution: {resolution}!\n'
                       f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
    if architecture_type not in _ARCHITECTURES_ALLOWED:
      raise ValueError(f'Invalid fused-scale option: {architecture_type}!\n'
                       f'Architectures allowed: {_ARCHITECTURES_ALLOWED}.')

    self.init_res = _INIT_RES
    self.resolution = resolution
    self.z_space_dim = z_space_dim
    self.w_space_dim = w_space_dim
    self.image_channels = image_channels
    self.architecture_type = architecture_type
    self.fused_modulate = fused_modulate
    self.truncation_psi = truncation_psi
    self.truncation_layers = truncation_layers
    self.randomize_noise = randomize_noise
    self.num_mapping_layers = num_mapping_layers
    self.fmaps_base = fmaps_base
    self.fmaps_max = fmaps_max

    self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2

    self.mapping = MappingModule(input_space_dim=self.z_space_dim,
                                 hidden_space_dim=self.fmaps_max,
                                 final_space_dim=self.w_space_dim,
                                 num_layers=self.num_mapping_layers,
                                 normalize_input=False)
    self.truncation = TruncationModule(num_layers=self.num_layers,
                                       w_space_dim=self.w_space_dim,
                                       truncation_psi=self.truncation_psi,
                                       truncation_layers=self.truncation_layers)
    self.synthesis = SynthesisModule(init_resolution=self.init_res,
                                     resolution=self.resolution,
                                     w_space_dim=self.w_space_dim,
                                     image_channels=self.image_channels,
                                     architecture_type=self.architecture_type,
                                     fused_modulate=self.fused_modulate,
                                     randomize_noise=self.randomize_noise,
                                     fmaps_base=self.fmaps_base,
                                     fmaps_max=self.fmaps_max)

    self.pth_to_tf_var_mapping = {}
    for key, val in self.mapping.pth_to_tf_var_mapping.items():
      self.pth_to_tf_var_mapping[f'mapping.{key}'] = val
    for key, val in self.truncation.pth_to_tf_var_mapping.items():
      self.pth_to_tf_var_mapping[f'truncation.{key}'] = val
    for key, val in self.synthesis.pth_to_tf_var_mapping.items():
      self.pth_to_tf_var_mapping[f'synthesis.{key}'] = val

  def forward(self, z):
    w = self.mapping(z)
    w = self.truncation(w)
    x = self.synthesis(w)
    return x
