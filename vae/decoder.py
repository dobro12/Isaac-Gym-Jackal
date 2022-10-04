from torch.nn import functional as F
from torch import jit, nn
import torch

class Decoder(jit.ScriptModule):
  def __init__(self, embedding_size=32, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc = nn.Linear(embedding_size, 1024)
    self.conv_transpose1 = nn.ConvTranspose2d(128, 128, 4, stride=2)  # 2x4x128   -> 6x10x128
    self.conv_transpose2 = nn.ConvTranspose2d(128, 64, 4, stride=2)   # 6x10x128  -> 14x22x64
    self.conv_transpose3 = nn.ConvTranspose2d(64, 32, 5, stride=2)    # 14x22x64  -> 31x47x32
    self.conv_transpose4 = nn.ConvTranspose2d(32, 1, 4, stride=2)     # 31x47x32  -> 64x96x1

  @jit.script_method
  def forward(self, latents):
    hidden = self.act_fn(self.fc(latents))
    hidden = hidden.view(-1, 128, 2, 4)
    hidden = self.act_fn(self.conv_transpose1(hidden))
    hidden = self.act_fn(self.conv_transpose2(hidden))
    hidden = self.act_fn(self.conv_transpose3(hidden))
    hidden = torch.sigmoid(self.conv_transpose4(hidden))
    return hidden
