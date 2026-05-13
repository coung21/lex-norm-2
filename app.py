import sys
import os
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import BARTphoMTL
from src.utils import load_checkpoint

import gradio as gr

# ── 1. Init System ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "artifacts/norm_checkpoint"
BARTPHO_PATH = os.path.join(CHECKPOINT_DIR, "bartpho")

if not os.path.exists(CHECKPOINT_DIR):
    print("❌ Lỗi: Không tìm thấy thư mục model tại", CHECKPOINT_DIR)
    print("Vui lòng chạy script Extrinsic Eval trước để tải model từ thư viện WandB.")
    sys.exit(1)

print("🚀 Khởi tạo Web UI...")
print(f"  [1/2] Loading Tokenizer từ {BARTPHO_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BARTPHO_PATH)

print(f"  [2/2] Loading Model BARTphoMTL từ {CHECKPOINT_DIR}...")
model = BARTphoMTL(model_name=BARTPHO_PATH, mode="mtl")
load_checkpoint(CHECKPOINT_DIR, model)
model = model.to(DEVICE)
model.eval()
print("✅ Hệ thống đã sẵn sàng Inference!")

# ── 2. Logic Inference ────────────────────────────────────────────────────────
def infer(text):
    if not text.strip():
        return [], ""
    
    # Chuẩn bị dữ liệu đầu vào
    inputs = tokenizer([text], max_length=128, truncation=True, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # Task 1: Detection
        det_preds = model.predict_detection(inputs["input_ids"], inputs["attention_mask"])
        
        # Task 2: Normalization
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )
        
    normalized_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    
    # ── Map Highlights ──
    # Gom token và nhãn hiển thị lên giao diện cực kỳ trực quan
    input_ids = inputs["input_ids"][0].cpu().tolist()
    pred_labels = det_preds[0].cpu().tolist()
    tokens_raw = tokenizer.convert_ids_to_tokens(input_ids)
    
    highlighted = []
    for tk, lbl in zip(tokens_raw, pred_labels):
        if tk in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
            
        # BARTpho dùng BPE nên các subword kết thúc bằng @@ 
        # Chúng ta gỡ mác @@ đi để dán chữ lên giao diện cho tự nhiên
        tk_clean = tk.replace("@@", "")
        label_str = "NSW (Lỗi)" if lbl == 1 else None
        
        highlighted.append((tk_clean + " ", label_str))
        
    return highlighted, normalized_text


# ── 3. Cấu Thiết Build Gradio Layout ──────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")) as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-weight: 900; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
            <span>🤖 Vietnamese Lexical Normalization System</span>
        </h1>
        <p style="font-size: 1.1rem; color: #6b7280; font-weight: 400;">
            Hệ thống phát hiện và chuẩn hóa Tiếng Việt.
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=4, 
                placeholder="Nhập câu tiếng việt có teencode của bạn...", 
                label="Văn bản cần quét (Input Text)",
                elem_classes="text-lg"
            )
            submit_btn = gr.Button("Chuẩn hóa văn bản", variant="primary", size="lg")
            
            
        with gr.Column(scale=1):
            # Panel hiển thị từ bị lỗi (Detection)
            det_output = gr.HighlightedText(
                show_legend=True, 
                label="NSW Detection", 
                color_map={"NSW (Lỗi)": "#ff4d4d"}
            )
            
            # Panel hiển thị sau chuẩn hóa (Normalization)
            norm_output = gr.Textbox(
                lines=4, 
                label="Lexical Normalization",
                elem_classes="text-lg font-bold"
            )
            
    # Bắt sự kiện
    submit_btn.click(fn=infer, inputs=input_text, outputs=[det_output, norm_output], api_name="normalize")
    input_text.submit(fn=infer, inputs=input_text, outputs=[det_output, norm_output])

if __name__ == "__main__":
    # Bật share=True để có link gửi cho thầy cô/bạn bè truy cập public
    demo.launch(server_name="0.0.0.0", share=False, server_port=7860)
