import os
import glob
import itertools
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

class SkullFaceDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, use_paired=False, use_explicit_pairs=False):
        self.dir_A = os.path.join(root_dir, f'{mode}A')
        self.dir_B = os.path.join(root_dir, f'{mode}B')
        self.transform = transform
        self.mode = mode
        self.use_paired = use_paired
        self.use_explicit_pairs = use_explicit_pairs
        self.root_dir = root_dir

        # Load explicit pairs if specified
        if use_explicit_pairs:
            pairs_file = os.path.join(root_dir, 'explicit_pairs.json')
            if os.path.exists(pairs_file):
                with open(pairs_file, 'r') as f:
                    self.explicit_pairs = json.load(f)
                self.size = len(self.explicit_pairs)
                print(f"Loaded {self.size} explicit pairs from {pairs_file}")
            else:
                raise FileNotFoundError(f"Explicit pairs file not found: {pairs_file}")
        else:
            # Original matching logic
            self.A_paths = sorted(glob.glob(os.path.join(self.dir_A, '*.png')))
            self.B_paths = sorted(glob.glob(os.path.join(self.dir_B, '*.png')))

            # Group B paths by (subject_id, view) for fast O(1) matching
            self.B_groups = {}
            for path in self.B_paths:
                filename = os.path.basename(path)
                parts = filename.split('_')
                if len(parts) >= 3:
                    subject_id = parts[1]  # e.g., '001'
                    view = parts[2]        # e.g., 'frontal' or 'lateral'
                    key = (subject_id, view)
                    if key not in self.B_groups:
                        self.B_groups[key] = []
                    self.B_groups[key].append(path)

            self.size = len(self.A_paths)

    def __len__(self):
        return self.size

    def apply_variant(self, img):
        # Choose a random variant from the 10 uploaded variants (0 to 9)
        variant = random.randint(0, 9)
        
        # Geometric/color parameters
        flip = False
        rotate_angle = 0.0
        contrast_factor = 1.0
        brightness_factor = 1.0
        erase = False
        
        if variant == 1:
            contrast_factor = random.uniform(0.8, 1.2)
        elif variant == 2:
            brightness_factor = random.uniform(0.85, 1.15)
        elif variant == 3:
            flip = True
        elif variant == 4:
            rotate_angle = random.uniform(-5, 5)
        elif variant == 5:
            erase = True
        elif variant == 6:
            contrast_factor = random.uniform(0.8, 1.2)
            rotate_angle = random.uniform(-5, 5)
        elif variant == 7:
            brightness_factor = random.uniform(0.85, 1.15)
        elif variant == 8:
            contrast_factor = random.uniform(0.8, 1.2)
            flip = True
        elif variant == 9:
            rotate_angle = random.uniform(-5, 5)
            brightness_factor = random.uniform(0.85, 1.15)
            erase = True

        if flip:
            img = TF.hflip(img)
        if rotate_angle != 0.0:
            img = TF.rotate(img, rotate_angle, fill=[0, 0, 0])
        if contrast_factor != 1.0:
            img = TF.adjust_contrast(img, contrast_factor)
        if brightness_factor != 1.0:
            img = TF.adjust_brightness(img, brightness_factor)
            
        return img, erase

    def __getitem__(self, index):
        # Use explicit pairs if specified
        if self.use_explicit_pairs:
            pair = self.explicit_pairs[index]
            A_path = os.path.join(self.dir_A, pair['skull'])
            B_path = os.path.join(self.dir_B, pair['face'])
        else:
            # Original matching logic
            A_path = self.A_paths[index % len(self.A_paths)]

            # Extract subject_id and view from A_path to find matching faces
            filename_A = os.path.basename(A_path)
            parts_A = filename_A.split('_')
            subject_id = parts_A[1]
            view = parts_A[2].replace('.png', '')
            key = (subject_id, view)

            # Select a matching face image of the same subject and view
            if key in self.B_groups and len(self.B_groups[key]) > 0:
                B_paths_subset = self.B_groups[key]
                B_index = random.randint(0, len(B_paths_subset) - 1)
                B_path = B_paths_subset[B_index]
            else:
                B_index = random.randint(0, len(self.B_paths) - 1)
                B_path = self.B_paths[B_index]

        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')

        erase_A = False
        erase_B = False
        if self.mode == 'train':
            # Apply dynamic on-the-fly variants independently
            img_A, erase_A = self.apply_variant(img_A)
            img_B, erase_B = self.apply_variant(img_B)

        # Apply basic scaling and tensor transforms
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        # Apply Random Erasing to tensors if needed
        if self.mode == 'train':
            erase_transform = transforms.RandomErasing(p=1.0, scale=(0.01, 0.05), ratio=(0.3, 3.3))
            if erase_A:
                img_A = erase_transform(img_A)
            if erase_B:
                img_B = erase_transform(img_B)

        return {'A': img_A, 'B': img_B}

# --- Models ---
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator, self).__init__()
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# --- Discriminator (Regularized with Dropout to stabilize unpaired geometry) ---
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True, dropout=0.0):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalize=False, dropout=0.0),
            *discriminator_block(64, 128, dropout=0.0),
            *discriminator_block(128, 256, dropout=0.2),
            *discriminator_block(256, 512, dropout=0.2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

# --- Training Code ---
def train():
    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Balanced Hyperparameters
    num_epochs = 1000
    batch_size = 16
    lr_G = 0.0002
    lr_D = 0.0002  # Use same LR - TTUR was causing instability

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use explicit pairs for all 1800 skull-face combinations (9 variations per skull)
    dataset = SkullFaceDataset('/storage2/shashank/gans/data/cyclegan_dataset', mode='train', transform=transform, use_explicit_pairs=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize models
    G_AB = Generator()
    G_BA = Generator()
    D_A = Discriminator()
    D_B = Discriminator()

    # Standard CycleGAN Loss Functions (Original)
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr_D, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr_D, betas=(0.5, 0.999))

    lambda_cyc = 10.0  # Standard CycleGAN weight
    lambda_id = 5.0    # Standard CycleGAN weight

    # Prepare for DDP with Accelerate
    G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, dataloader = accelerator.prepare(
        G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, dataloader
    )

    if accelerator.is_main_process:
        print(f"Starting stabilized + 10x augmented CycleGAN training on {device}...")
    
    # Tracking losses for plotting
    history_loss_G = []
    history_loss_D = []
    
    for epoch in range(num_epochs):
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
            pbar = dataloader

        for i, batch in enumerate(pbar):
            real_A = batch['A']
            real_B = batch['B']

            # Receptive field target sizes
            valid = torch.ones((real_A.size(0), 1, 16, 16), device=device)
            fake = torch.zeros((real_A.size(0), 1, 16, 16), device=device)

            # Soft label smoothing (no random noise - that was destabilizing)
            valid_smooth = torch.full((real_A.size(0), 1, 16, 16), 0.95, device=device)
            fake_smooth = torch.full((real_A.size(0), 1, 16, 16), 0.05, device=device)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A) * lambda_id
            loss_id_B = criterion_identity(G_AB(real_B), real_B) * lambda_id

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)

            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A) * lambda_cyc

            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B) * lambda_cyc

            # Total loss - Standard CycleGAN (Original)
            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            accelerator.backward(loss_G)
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid_smooth)
            loss_fake = criterion_GAN(D_A(fake_A.detach()), fake_smooth)
            loss_D_A = (loss_real + loss_fake) / 2
            accelerator.backward(loss_D_A)
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid_smooth)
            loss_fake = criterion_GAN(D_B(fake_B.detach()), fake_smooth)
            loss_D_B = (loss_real + loss_fake) / 2
            accelerator.backward(loss_D_B)
            optimizer_D_B.step()
            
            # Gather losses across all processes for averaging
            gathered_loss_G = accelerator.gather_for_metrics(loss_G).mean().item()
            gathered_loss_D = accelerator.gather_for_metrics(loss_D_A + loss_D_B).mean().item()

            epoch_loss_G += gathered_loss_G
            epoch_loss_D += gathered_loss_D

            if accelerator.is_main_process:
                pbar.set_postfix({"loss_G": gathered_loss_G, "loss_D": gathered_loss_D})
            
        # Record and save average epoch loss
        if accelerator.is_main_process:
            history_loss_G.append(epoch_loss_G / len(dataloader))
            history_loss_D.append(epoch_loss_D / len(dataloader))

            with open("loss_history.json", "w") as f:
                json.dump({"loss_G": history_loss_G, "loss_D": history_loss_D}, f)
                
            # Save checkpoints
            if (epoch + 1) % 50 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                
                unwrapped_G_AB = accelerator.unwrap_model(G_AB)
                unwrapped_G_BA = accelerator.unwrap_model(G_BA)
                
                torch.save(unwrapped_G_AB.state_dict(), f"checkpoints/G_AB_{epoch+1}.pth")
                torch.save(unwrapped_G_BA.state_dict(), f"checkpoints/G_BA_{epoch+1}.pth")
                
                prev_epoch = (epoch + 1) - 50
                if prev_epoch > 0:
                    for prefix in ["G_AB", "G_BA"]:
                        old_ckpt = f"checkpoints/{prefix}_{prev_epoch}.pth"
                        if os.path.exists(old_ckpt):
                            os.remove(old_ckpt)

if __name__ == '__main__':
    train()
