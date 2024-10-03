from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 0.001
    cuda_num: int = 0
    dtype: torch.dtype = torch.float32
    momentum: float = 0.9
    device: torch.device = torch.device('mps')
    image_resize: tuple = (32, 32)