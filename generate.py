import torch

def generate_music(model, tokenizer, config, start_tokens=None, output_path="generated_song.mid"):
    if model is None or tokenizer is None:
        return

    if start_tokens is None:
        start_tokens = torch.full(
            (1, 1), tokenizer["BOS_None"], dtype=torch.long, device=config.device
        )

    generated_ids = model.generate(start_tokens, max_new_tokens=config.max_new_tokens)
    tokens = generated_ids[0].tolist()

    score = tokenizer.decode([tokens])
    score.dump_midi(output_path)
    print(f"Music generated and saved to {output_path}")