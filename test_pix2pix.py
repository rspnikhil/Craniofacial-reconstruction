import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
from train_pix2pix import GeneratorUNet

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = GeneratorUNet().to(device)
    checkpoint_path = "checkpoints/G_1000.pth" 
    
    if os.path.exists(checkpoint_path):
        generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with untrained weights.")

    generator.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    inverse_transform = transforms.Compose([
        transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
        transforms.ToPILImage()
    ])

    os.makedirs("test_outputs", exist_ok=True)
    test_dir = "/storage2/shashank/gans/data/pix2pix_dataset/test"
    all_paths = sorted(glob.glob(os.path.join(test_dir, '*.png')))
    image_paths = [
        p for p in all_paths
        if "FS_to_FF" in os.path.basename(p) or "SS_to_LF" in os.path.basename(p)
    ]

    processed = set()

    for path in image_paths:
        filename = os.path.basename(path)
        parts = filename.split('_')
        if len(parts) >= 3:
            subject_id = parts[1]  # e.g., '115'
            view = parts[2]        # e.g., 'FS' or 'SS'
            key = (subject_id, view)
            if key in processed:
                continue
            processed.add(key)

            view_name = "frontal" if view == "FS" else "lateral"

            img = Image.open(path).convert('RGB')
            w, h = img.size
            w2 = int(w / 2)
            img_A = img.crop((0, 0, w2, h))

            input_tensor = transform(img_A).unsqueeze(0).to(device)

            with torch.no_grad():
                output_tensor = generator(input_tensor)

            output_img = inverse_transform(output_tensor.squeeze(0).cpu())
            out_filename = f"predicted_face_subject_{subject_id}_{view_name}.png"
            output_img.save(f"test_outputs/{out_filename}")
            print(f"Saved prediction for subject {subject_id} ({view_name}) to test_outputs/{out_filename}")

if __name__ == '__main__':
    test()
