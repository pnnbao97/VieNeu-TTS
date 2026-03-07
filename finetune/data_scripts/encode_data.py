import os
import torch
import librosa
import json
from tqdm import tqdm
from neucodec import NeuCodec

import random

def encode_dataset(dataset_dir="finetune/dataset", max_samples=2000):
    metadata_path = os.path.join(dataset_dir, "metadata_cleaned.csv")
    if not os.path.exists(metadata_path):
        print("🦜 Không tìm thấy metadata_cleaned.csv, thử dùng metadata.csv gốc...")
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        
    output_path = os.path.join(dataset_dir, "metadata_encoded.csv")
    raw_audio_dir = os.path.join(dataset_dir, "raw_audio")

    if not os.path.exists(metadata_path):
        print("🦜 Không tìm thấy file metadata nào!")
        return

    print("🦜 Đang tải NeuCodec model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
    codec.eval()

    print(f"🦜 Bắt đầu encode metadata: {metadata_path}")
    print(f"🦜 Đang lấy ngẫu nhiên tối đa {max_samples} mẫu để xử lý (bạn có thể chỉnh sửa số lượng này trong code).")
    
    lines_to_write = []
    skipped_count = 0
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Shuffle ngẫu nhiên và lấy max_samples
    random.shuffle(lines)
    if len(lines) > max_samples:
        lines = lines[:max_samples]
        
    for line in tqdm(lines):
        parts = line.strip().split('|')
        if len(parts) < 2: 
            continue
        
        filename = parts[0]
        text = parts[1]
        
        audio_path = os.path.join(raw_audio_dir, filename)
        
        if not os.path.exists(audio_path):
            skipped_count += 1
            continue
            
        try:
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)

            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                codes = codec.encode_code(wav_tensor)
                
                codes = codes.squeeze(0).squeeze(0).cpu().numpy().flatten().tolist()
                codes = [int(x) for x in codes]
            
            # Validate codes
            if not codes or len(codes) == 0:
                print(f"🦜 Empty codes cho file: {filename}")
                skipped_count += 1
                continue
            
            if not all(0 <= c < 65536 for c in codes):
                print(f"🦜 Invalid code range cho file: {filename}")
                skipped_count += 1
                continue
            
            codes_json = json.dumps(codes)
            
            # Format output: filename|text|codes
            lines_to_write.append(f"{filename}|{text}|{codes_json}\n")
            
        except Exception as e:
            print(f"🦜 Lỗi xử lý file {filename}: {e}")
            skipped_count += 1
            
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines_to_write)
        
    print(f"\n🦜 Hoàn tất! Đã lưu file mã hóa tại: {output_path}")
    print(f"   - Tổng file xử lý thành công: {len(lines_to_write)}")
    print(f"   - Số file lỗi/bỏ qua: {skipped_count}")

if __name__ == "__main__":
    # Luôn xác định đường dẫn tương đương với thư mục gốc của project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    target_dir = os.path.join(project_root, "finetune", "dataset")
    
    encode_dataset(dataset_dir=target_dir)
