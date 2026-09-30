import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from train_cyclegan import Generator
import numpy as np
import json

def compute_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Generators and load weights
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)

    checkpoint_AB = "checkpoints/G_AB_1000.pth"
    checkpoint_BA = "checkpoints/G_BA_1000.pth"

    if os.path.exists(checkpoint_AB):
        G_AB.load_state_dict(torch.load(checkpoint_AB, map_location=device))
    else:
        print(f"Warning: Checkpoint {checkpoint_AB} not found.")

    if os.path.exists(checkpoint_BA):
        G_BA.load_state_dict(torch.load(checkpoint_BA, map_location=device))
    else:
        print(f"Warning: Checkpoint {checkpoint_BA} not found.")

    G_AB.eval()
    G_BA.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    inverse_transform = transforms.Compose([
        transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)), # Un-normalize
        transforms.ToPILImage()
    ])

    os.makedirs("test_outputs", exist_ok=True)

    import glob
    test_dir = "/storage2/shashank/gans/data/cyclegan_dataset/testA"
    skull_paths = sorted(glob.glob(os.path.join(test_dir, "*.png")))

    psnr_scores = []

    for path in skull_paths:
        if not os.path.exists(path):
            continue

        filename = os.path.basename(path)
        img = Image.open(path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # Generate face from skull
            fake_face = G_AB(input_tensor)
            # Reconstruct skull from face (cycle consistency)
            reconstructed_skull = G_BA(fake_face)
            # Calculate PSNR of cycle consistency
            psnr = compute_psnr(reconstructed_skull, input_tensor, max_val=1.0)
            psnr_scores.append(psnr)

        output_img = inverse_transform(fake_face.squeeze(0).cpu())
        output_img.save(f"test_outputs/predicted_face_{filename}")
        print(f"Saved prediction for {filename} | PSNR (cycle): {psnr:.4f} dB")

    # Save PSNR metrics
    if psnr_scores:
        avg_psnr = np.mean(psnr_scores)
        std_psnr = np.std(psnr_scores)
        max_psnr = np.max(psnr_scores)
        min_psnr = np.min(psnr_scores)

        metrics = {
            "avg_psnr": float(avg_psnr),
            "std_psnr": float(std_psnr),
            "max_psnr": float(max_psnr),
            "min_psnr": float(min_psnr),
            "num_samples": len(psnr_scores)
        }

        with open("test_outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "="*50)
        print("CycleGAN Test Metrics (Cycle Consistency PSNR)")
        print("="*50)
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Std Dev PSNR: {std_psnr:.4f} dB")
        print(f"Max PSNR: {max_psnr:.4f} dB")
        print(f"Min PSNR: {min_psnr:.4f} dB")
        print(f"Number of samples: {len(psnr_scores)}")
        print("="*50)

if __name__ == '__main__':
    test()
