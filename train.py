import torch
from model import MusicGPT

def train_model(midi_folder_path, tokenizer, config, build_dataloader_fn):
    loader, midi_paths = build_dataloader_fn(
        midi_folder_path, tokenizer, config.block_size, config.batch_size
    )
    if loader is None:
        print(f"\n❌ ERROR: No MIDI files found in folder: '{midi_folder_path}'")
        return None

    model = MusicGPT(
        vocab_size=len(tokenizer),
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
        device=config.device,
    ).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    print(f"Starting training on {len(midi_paths)} files...")
    for epoch in range(config.epochs):
        total_loss = 0.0
        for batch in loader:
            x = batch["input_ids"].to(config.device)
            y = batch["labels"].to(config.device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} complete. Avg Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), config.model_path)
    print(f"Model saved as {config.model_path}")
    return model