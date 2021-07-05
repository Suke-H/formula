import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F

"""
hparams:
  patch_size: 25
  hidden_n: 64
  feature_n: 2
  output_n: ${data.class_num}
  pool_mode: sum
  patch_num_min: 1
  patch_num_max: 64
  seed: 0
  batch_size: 256
  num_workers: 6
  max_epochs: 100000
  min_epochs: 10
  patience: 100
  optimizer: Adam
  lr: 0
  data_split_num: ${data.datamodule.data_split_num}
  data_use_num: ${data.datamodule.data_use_num}

"""

class DeepSets(nn.Module):
    def __init__(self, patch_size, feature_n, output_n, pool_mode):
        super().__init__()

        self.f_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size ** 2, 64),
            nn.ReLU(),
            nn.Linear(64, feature_n),
        )

        self.f = nn.Sequential(
            nn.Linear(feature_n, 64),
            nn.ReLU(),
            nn.Linear(64, output_n),
        )

        self.pool_mode = pool_mode

    def encode(self, patch_set):
        if isinstance(patch_set, Tensor):
            x = patch_set.reshape(
                patch_set.shape[0] * patch_set.shape[1], *patch_set.shape[2:]
            )
            x = self.f_1(x)
            x = x.reshape(*patch_set.shape[:2], *x.shape[1:])
            
        elif isinstance(patch_set, list):
            x = patch_set
            x = [self.f_1(x) for x in x]

        else:
            assert False

        return x

    def pool(self, feature_set, keepdim=False):
        if self.pool_mode == 'max':
            pool = lambda *args, **kwargs,: torch.max(*args, **kwargs)[0]
        else:
            pool = getattr(torch, self.pool_mode)

        if isinstance(feature_set, Tensor):
            x = pool(feature_set, 1, keepdim=keepdim)
        elif isinstance(feature_set, list):
            x = [pool(i, 0, keepdim=keepdim) for i in feature_set]
            x = torch.stack(x)
        else:
            assert False

        return x

    def decode(self, feature):
        x = feature
        x = self.f(x)

        return x

    def forward(self, patch_set):
        feature_set = self.encode(patch_set)
        features = self.pool(feature_set)
        output = self.decode(features)

        return output
