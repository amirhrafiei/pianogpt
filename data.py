from pathlib import Path
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader

def get_tokenizer():
    tokenizer_config = TokenizerConfig(num_velocities=16, use_chords=True, use_tempos=True)
    return REMI(tokenizer_config)

def build_dataloader(midi_folder_path, tokenizer, block_size, batch_size):
    midi_paths = []
    for ext in ["*.mid", "*.midi"]:
        midi_paths.extend(list(Path(midi_folder_path).glob(f"**/{ext}")))

    if not midi_paths:
        return None, []

    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=block_size,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )

    collator = DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True, shift_labels=True)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    return loader, midi_paths