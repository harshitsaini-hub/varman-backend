import torch

def system_check():
    print("\n--- Varman MVP Hardware Diagnostics ---")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✅ GPU Detected: {gpu_name}")
        print(f"✅ Total VRAM: {vram_total:.2f} GB")
        print("STATUS: Ready for Heavy PyTorch Math.")
    else:
        print("❌ GPU Not Found! PyTorch is defaulting to CPU.")
        print("STATUS: Needs CUDA installation check before proceeding.")
        
if __name__ == "__main__":
    system_check()