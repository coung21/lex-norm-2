from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.normalizers import NFC
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

CORPUS = ["data/tokenizer_corpus/tokenizer_corpus.txt"]
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
VOCAB_SIZE = 16000
MIN_FREQ = 2

def save_tokenizer(tokenizer, save_dir):
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))

def add_postprocess(tokenizer):
    tokenizer.post_processor = TemplateProcessing(
        single=f"<s> $A </s>",
        pair=f"<s> $A </s> </s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
            
        ])
    return tokenizer

def train_bpe():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = BPEDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(files=CORPUS, trainer=trainer)
    tokenizer = add_postprocess(tokenizer)
    save_tokenizer(tokenizer, "tokenizers/bpe")


def train_byte_bpe():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(files=CORPUS, trainer=trainer)
    tokenizer = add_postprocess(tokenizer)
    save_tokenizer(tokenizer, "tokenizers/byte_bpe")

    
if __name__ == "__main__":
    train_bpe()
    train_byte_bpe()