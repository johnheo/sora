import os

from src.utils import import_modules
from omegaconf import OmegaConf
from torch import nn
try:
    import esm
    ESM_INSTALLED = True
except:
    ESM_INSTALLED = False

MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

# automatically import any Python files in the models/ directory
import_modules(os.path.dirname(__file__), "src.models")


class FixedBackboneDesignEncoderDecoder(nn.Module):
    _default_cfg = {}

    def __init__(self, cfg) -> None:
        super().__init__()
        self._update_cfg(cfg)

    def _update_cfg(self, cfg):
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)

    @classmethod
    def from_config(cls, cfg):
        raise NotImplementedError

    def forward_encoder(self, batch):
        raise NotImplementedError

    def forward_decoder(self, prev_decoder_out, encoder_out):
        raise NotImplementedError

    def initialize_output_tokens(self, batch, encoder_out):
        raise NotImplementedError

    def forward(self, coords, coord_mask, tokens, token_padding_mask=None, **kwargs):
        raise NotImplementedError

    def sample(self, coords, coord_mask, tokens=None, token_padding_mask=None, **kwargs):
        raise NotImplementedError
