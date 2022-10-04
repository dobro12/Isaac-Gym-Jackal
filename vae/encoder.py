from torch.nn import functional as F
from torch import jit, nn
import torch

class Encoder(jit.ScriptModule):
  def __init__(self, embedding_size=32, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.conv1 = nn.Conv2d(1, 32, 4, stride=2)      # 64x96x1   -> 31x47x32
    self.conv2 = nn.Conv2d(32, 64, 5, stride=2)     # 31x47x32  -> 14x22x64
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)    # 14x22x64  -> 6x10x128
    self.conv4 = nn.Conv2d(128, 128, 4, stride=2)   # 6x10x128  -> 2x4x128 = 1024
    self.fc_mean = nn.Linear(1024, embedding_size)
    self.fc_std = nn.Linear(1024, embedding_size)

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = hidden.view(-1, 1024)
    mean = self.fc_mean(hidden)
    log_var = self.fc_std(hidden)
    std = torch.exp(log_var/2.0)
    return mean, log_var, std
