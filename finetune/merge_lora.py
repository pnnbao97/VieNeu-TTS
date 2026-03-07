import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def merge_lora(base_model_path, adapter_path, output_path):
    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", type=str, required=True, help="Base model HuggingFace ID or local path")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter checkout")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")

    args = parser.parse_args()
    merge_lora(args.base_model, args.adapter, args.output)
