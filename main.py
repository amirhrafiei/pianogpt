import os
import torch
from config import TrainConfig
from data import get_tokenizer, build_dataloader
from model import MusicGPT
from train import train_model
from generate import generate_music

def main():
    config = TrainConfig()
    tokenizer = get_tokenizer()

    if os.path.exists(config.model_path):
        print(f"Found existing model at '{config.model_path}'. Loading...")
        model = MusicGPT(
            vocab_size=len(tokenizer),
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            block_size=config.block_size,
            dropout=config.dropout,
            device=config.device,
        ).to(config.device)
        state = torch.load(config.model_path, map_location=config.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("⚠️ Checkpoint mismatch. Some weights were not loaded.")
            print("Missing:", missing)
            print("Unexpected:", unexpected)
    else:
        print("No saved model found. Starting fresh training...")
        model = train_model("my_midis", tokenizer, config, build_dataloader)

    if model is not None:
        print("Composing new melodies...")
        for i in range(3):
            generate_music(model, tokenizer, config, output_path=f"ai_composition_{i+1}.mid")
    else:
        print("Execution halted. Check your data and environment.")

if __name__ == "__main__":
    main()