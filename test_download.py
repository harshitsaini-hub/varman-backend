import requests
import sys

def verify_secure_image_fetch(image_id: str, jwt_token: str, save_path: str):
    print(f"Initiating secure fetch for asset: {image_id}...")
    url = f"http://localhost:8000/api/images/download/{image_id}"
    headers = {"Authorization": f"Bearer {jwt_token}"}

    try:
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"✅ Blob successfully extracted to {save_path}")
        else:
            print(f"❌ Decryption rejected. Status: {response.status_code}")
            print(f"Response detail: {response.text}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_download.py <image_uuid> <jwt_token> [save_path.jpg]")
        sys.exit(1)
        
    image_uuid = sys.argv[1]
    token = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "verification_output.jpg"
    
    verify_secure_image_fetch(image_uuid, token, out_path)
