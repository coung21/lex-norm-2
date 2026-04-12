import pandas as pd 
from argparse import ArgumentParser
from pathlib import Path

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/tokenizer_corpus")
    args = parser.parse_args()

    OUT = Path(args.output_path)
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)

    for idx, row in df.iterrows():
        text = row["text"]

        with open(OUT / "tokenizer_corpus.txt", "a", encoding="utf-8") as f:
            f.write(text + "\n")
        
    print(f"Done! Saved to {OUT / 'tokenizer_corpus.txt'}")

if __name__ == "__main__":
    main()