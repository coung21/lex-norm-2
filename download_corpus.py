from datasets import load_dataset
from pathlib import Path

# Load dataset
dataset = load_dataset("levuloihust/vien-corpus-for-tokenizer", split="train")

# Define output path
out_dir = Path("data/tokenizer_corpus")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "vien_corpus.txt"

print(f"Downloading and generating {out_file}...")

with open(out_file, "w", encoding="utf-8") as f:
    # Check column names. Usually it's 'text'
    if 'text' in dataset.column_names:
        text_col = 'text'
    else:
        text_col = dataset.column_names[0]
        
    for item in dataset:
        f.write(item[text_col] + "\n")

print("Done generating corpus!")
