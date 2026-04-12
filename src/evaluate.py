import torch
from transformers import PreTrainedTokenizerFast
import math
import sys
import os

from build_dataset import NormDataset
from torch.utils.data import DataLoader
from model import Seq2SeqTransformer
import wandb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def edit_distance(ref, hyp):
    if len(ref) > len(hyp):
        ref, hyp = hyp, ref
    distances = range(len(ref) + 1)
    for i2, c2 in enumerate(hyp):
        distances_ = [i2+1]
        for i1, c1 in enumerate(ref):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def calculate_wer(ref, hyp):
    return edit_distance(ref.split(), hyp.split())

def calculate_cer(ref, hyp):
    return edit_distance(list(ref), list(hyp))

class Evaluator:
    def __init__(self, model_path, tokenizer_dir, dev_path, batch_size=32):
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{tokenizer_dir}/tokenizer.json",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
        )
        self.model = Seq2SeqTransformer(
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id
        ).to(DEVICE)
        
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        
        self.dev_ds = NormDataset(dev_path, self.tokenizer)
        self.dev_dl = DataLoader(self.dev_ds, batch_size=batch_size, shuffle=False)

    def greedy_decode(self, src_ids, max_len=128):
        batch_size = src_ids.size(0)
        src_key_padding_mask = src_ids.eq(self.tokenizer.pad_token_id)
        src = self.model.pos_encoder(self.model.embed(src_ids) * math.sqrt(self.model.d_model))
        memory = self.model.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=DEVICE)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

        for _ in range(max_len - 1):
            tgt_len = ys.size(1)
            tgt_mask = self.model.transformer.generate_square_subsequent_mask(tgt_len).to(DEVICE)
            tgt = self.model.pos_encoder(self.model.embed(ys) * math.sqrt(self.model.d_model))
            out = self.model.transformer.decoder(
                tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask
            )
            prob = self.model.fc_out(out[:, -1, :])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            finished |= (next_word == self.tokenizer.eos_token_id)
            if finished.all():
                break
        return ys

    def evaluate(self):
        total_exact_match = 0
        total_cer = 0
        total_wer_model = 0
        total_wer_baseline = 0
        total_gt_words = 0
        total_gt_chars = 0
        total_samples = 0
        
        # Tạo bảng logging cho WandB
        table = None
        if wandb.run is not None:
            table = wandb.Table(columns=["Original", "Ground Truth", "Prediction", "Correct?"])

        print("Starting Evaluation and Logging Inference...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dev_dl):
                src_ids = torch.tensor(batch['input_ids'], device=DEVICE)
                pred_ids = self.greedy_decode(src_ids)
                
                for idx in range(src_ids.size(0)):
                    dataset_idx = batch_idx * self.dev_dl.batch_size + idx
                    ex = self.dev_ds.data.iloc[dataset_idx]
                    
                    original_text = str(ex['original']).strip()
                    ground_truth_text = str(ex['normalized']).strip()
                    
                    p_list = pred_ids[idx].tolist()
                    if self.tokenizer.eos_token_id in p_list:
                        eos_idx = p_list.index(self.tokenizer.eos_token_id)
                        p_list = p_list[:eos_idx]
                    p_list = [t for t in p_list if t not in [self.tokenizer.pad_token_id, self.tokenizer.bos_token_id]]
                    predict_text = self.tokenizer.decode(p_list).strip()
                    
                    is_correct = (predict_text == ground_truth_text)
                    if is_correct:
                        total_exact_match += 1
                        
                    # Log vào WandB Table (giới hạn log 200 câu đầu hoặc ngẫu nhiên để view nhanh)
                    if table is not None and total_samples < 200:
                        table.add_data(original_text, ground_truth_text, predict_text, "✅" if is_correct else "❌")
                    
                    total_wer_baseline += calculate_wer(ground_truth_text, original_text)
                    total_wer_model += calculate_wer(ground_truth_text, predict_text)
                    total_cer += calculate_cer(ground_truth_text, predict_text)
                    
                    total_gt_words += len(ground_truth_text.split())
                    total_gt_chars += len(list(ground_truth_text))
                    total_samples += 1

                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed samples: {total_samples}")

        # Tính toán kết quả cuối
        em_rate = total_exact_match / total_samples
        cer = total_cer / max(1, total_gt_chars)
        wer = total_wer_model / max(1, total_gt_words)
        err = (total_wer_baseline - total_wer_model) / total_wer_baseline if total_wer_baseline > 0 else 0.0
            
        print("="*40)
        print(f"EVALUATION RESULTS ({total_samples} samples)")
        print(f"EM: {em_rate*100:.2f}% | WER: {wer*100:.2f}% | CER: {cer*100:.2f}% | ERR: {err*100:.2f}%")
        
        if wandb.run is not None:
            wandb.log({
                "eval/exact_match": em_rate * 100,
                "eval/wer": wer * 100,
                "eval/cer": cer * 100,
                "eval/err": err * 100,
                "eval/predictions": table
            })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--dev_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--run_name", type=str, default="eval_inference_log")
    args = parser.parse_args()

    wandb.init(project="lexnorm2", name=args.run_name, job_type="evaluation")
    evaluator = Evaluator(args.model_path, args.tokenizer_dir, args.dev_path, args.batch_size)
    evaluator.evaluate()
    wandb.finish()
