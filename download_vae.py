import os
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

def download_vae():
    model_id = "stabilityai/sd-vae-ft-mse"
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sd-vae-ft-mse")
    
    print(f"Downloading {model_id}...")
    vae = AutoencoderKL.from_pretrained(model_id)
    
    print(f"Saving to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    vae.save_pretrained(save_path)
    
    print("Download complete. Model is now stored locally.")

if __name__ == "__main__":
    download_vae()
