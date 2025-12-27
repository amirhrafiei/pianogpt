import torch
import torch.nn as nn
from torch.nn import functional as F
from miditok import REMI, TokenizerConfig
from symusic import Score
from pathlib import Path
from torch.utils.data import DataLoader
from miditok.pytorch_data import DatasetMIDI, DataCollator
import os

# --- 1. CONFIGURATION ---
batch_size = 16
block_size = 256
n_embd = 128
n_head = 4
n_layer = 4
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "music_gpt_model.pth"

# --- 2. THE BRAIN PIECES ---

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 3. THE MAIN MODEL ---

class MusicGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx) 
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, C), targets.reshape(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        self.eval() 
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 4. DATA PIPELINE & TRAINING ---

def get_tokenizer():
    tokenizer_config = TokenizerConfig(num_velocities=16, use_chords=True, use_tempos=True)
    return REMI(tokenizer_config)

def train_model(midi_folder_path):
    tokenizer = get_tokenizer()
    
    midi_paths = []
    for ext in ['*.mid', '*.midi']:
        midi_paths.extend(list(Path(midi_folder_path).glob(f"**/{ext}")))
        
    if not midi_paths:
        print(f"\n❌ ERROR: No MIDI files found in folder: '{midi_folder_path}'")
        return None, None 

    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=block_size,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )

    collator = DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True, shift_labels=True)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    
    model = MusicGPT(vocab_size=len(tokenizer)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    print(f"Starting training on {len(midi_paths)} files...")
    for epoch in range(50):
        total_loss = 0
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} complete. Avg Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")
    return model, tokenizer

def generate_music(model, tokenizer, start_tokens=None, output_path="generated_song.mid"):
    if model is None or tokenizer is None:
        return
        
    if start_tokens is None:
        start_tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    generated_ids = model.generate(start_tokens, max_new_tokens=500)
    tokens = generated_ids[0].tolist()
    
    score = tokenizer.decode([tokens])
    score.dump_midi(output_path)
    print(f"Music generated and saved to {output_path}")

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    
    # NEW: Check if a trained brain already exists
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at '{MODEL_PATH}'. Loading...")
        model = MusicGPT(vocab_size=len(tokenizer)).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("No saved model found. Starting fresh training...")
        model, tokenizer = train_model("my_midis")
    
    if model is not None:
        print("Composing new melodies...")
        # Generate 3 different songs for variety
        for i in range(3):
            generate_music(model, tokenizer, output_path=f"ai_composition_{i+1}.mid")
    else:
        print("Execution halted. Check your data and environment.")