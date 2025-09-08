# Copyright (c) CAIRI AI Lab. All rights reserved

from .fsae import FSAE
from .fsae_ms_predictor import FSAEMSPredictor

method_maps = {
    'fsae': FSAE,
    'fsaemspredictor': FSAEMSPredictor,
}

__all__ = [
    'FSAE', 'FSAEMSPredictor',
]