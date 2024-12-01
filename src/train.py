import os
import subprocess
from PIL import Image
import glob

# File paths
STYLEGAN2_PATH = os.path.join(os.getcwd(), "stylegan2")
DATASET_PATH = os.path.join(os.getcwd(), "data", "images")
OUTPUT_PATH = os.path.join(os.getcwd(), "models", "results")

# Training configuration
CONFIG = {
    "data": DATASET_PATH,
    "outdir": OUTPUT_PATH,
    "gpus": 1,
    "batch_size": 16,
    "gamma": 15,
    "snap": 10,
}

# Helper function to preprocess training data (images)
def preprocess_images(dataset_path, target_size=(128, 128)):
    print("Preprocessing images...")
    processed_dir = os.path.join(dataset_path, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    for img_path in glob.glob(os.path.join(dataset_path, "*.png")) + glob.glob(
        os.path.join(dataset_path, "*.jpg")
    ):
        try:
            img = Image.open(img_path).convert("RGB").resize(target_size)
            img_name = os.path.basename(img_path)
            img.save(os.path.join(processed_dir, img_name))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("Preprocessing completed; results saved at", processed_dir)
    return processed_dir


def train_stylegan(config):
    os.makedirs(config["outdir"], exist_ok=True)
    train_command = [
        "python",
        os.path.join(STYLEGAN2_PATH, "train.py"),
        f"--outdir={config['outdir']}",
        f"--data={config['data']}",
        f"--gpus={config['gpus']}",
        "--cfg=auto",  # automatically configure training
        f"--batch={config['batch_size']}",
        f"--gamma={config['gamma']}",
        f"--snap={config['snap']}",
    ]

    print("Starting StyleGAN2 training...")
    try:
        subprocess.run(train_command, check=True)
        print("Training completed! Hurray!")
    except subprocess.CalledProcessError as e:
        print("Error during training :(", e)

# main
if __name__ == "__main__":
    processed_data_path = preprocess_images(DATASET_PATH)
    CONFIG["data"] = processed_data_path
    train_stylegan(CONFIG)
