from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    block_size: int = 256
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    learning_rate: float = 3e-4
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: str = "music_gpt_model.pth"
    epochs: int = 50
    max_new_tokens: int = 500