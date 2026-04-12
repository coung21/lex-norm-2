from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizers/bpe/tokenizer.json",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
)

ids = tokenizer.encode("xin chào các bạn")
print("Encoded:", ids)
print("Decoded:", tokenizer.decode(ids, skip_special_tokens=True))
print("Tokens:", tokenizer.convert_ids_to_tokens(ids))
