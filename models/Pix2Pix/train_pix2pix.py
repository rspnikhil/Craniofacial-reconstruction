import os
import glob
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

class Pix2PixDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.dir = os.path.join(root_dir, mode)
        all_paths = sorted(glob.glob(os.path.join(self.dir, '*.png')))
        # Filter for anatomically aligned pairs to avoid contradictory one-to-many mappings (ghosting)
        self.image_paths = [
            p for p in all_paths
            if "FS_to_FF" in os.path.basename(p) or "SS_to_LF" in os.path.basename(p)
        ]
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return self.image_paths.__len__()

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        
        w, h = img.size
        w2 = int(w / 2)
        
        # Left side is skull (domain A), Right side is face (domain B)
        img_A = img.crop((0, 0, w2, h))
        img_B = img.crop((w2, 0, w, h))

        erase = False
        if self.mode == 'train':
            # Choose a random variant from the 10 uploaded variants (0 to 9)
            variant = random.randint(0, 9)
            
            # Geometric/color parameters
            flip = False
            rotate_angle = 0.0
            contrast_factor = 1.0
            brightness_factor = 1.0
            
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

            # Apply identical geometric transforms to keep Domain A and B spatially aligned
            if flip:
                img_A = TF.hflip(img_A)
                img_B = TF.hflip(img_B)
            if rotate_angle != 0.0:
                img_A = TF.rotate(img_A, rotate_angle, fill=[0, 0, 0])
                img_B = TF.rotate(img_B, rotate_angle, fill=[0, 0, 0])

            # Apply identical color transforms to preserve corresponding features
            if contrast_factor != 1.0:
                img_A = TF.adjust_contrast(img_A, contrast_factor)
                img_B = TF.adjust_contrast(img_B, contrast_factor)
            if brightness_factor != 1.0:
                img_A = TF.adjust_brightness(img_A, brightness_factor)
                img_B = TF.adjust_brightness(img_B, brightness_factor)

        # Apply basic scaling and tensor transforms
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        # Random Erasing (on tensor, up to 5% of area)
        if self.mode == 'train' and erase:
            erase_transform = transforms.RandomErasing(p=1.0, scale=(0.01, 0.05), ratio=(0.3, 3.3))
            img_A = erase_transform(img_A)
            img_B = erase_transform(img_B)

        return {'A': img_A, 'B': img_B, 'path': img_path}

# --- UNet Generator ---
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=False)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)

# --- PatchGAN Discriminator (Weighed down with Dropout to balance competition) ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalization=True, dropout=0.3):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1, bias=False)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, stride=2, normalization=False, dropout=0.0),
            *discriminator_block(64, 128, stride=2, dropout=0.3),
            *discriminator_block(128, 256, stride=2, dropout=0.3),
            *discriminator_block(256, 512, stride=1, dropout=0.3),
            nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=True)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# Weight Initialization
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

# --- Training Code ---
def train():
    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Balanced Hyperparameters
    num_epochs = 1000
    batch_size = 16
    lr_G = 0.0002
    lr_D = 0.00002  # TTUR: 10x slower D
    lambda_pixel = 30
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = Pix2PixDataset('/storage2/shashank/gans/data/pix2pix_dataset', mode='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Models
    generator = GeneratorUNet()
    discriminator = Discriminator()

    # Apply standard normal weight initialization
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Standard BCE Loss for conditional GAN
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixelwise = nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))

    # Prepare for DDP with Accelerate
    generator, discriminator, optimizer_G, optimizer_D, dataloader = accelerator.prepare(
        generator, discriminator, optimizer_G, optimizer_D, dataloader
    )

    if accelerator.is_main_process:
        print(f"Starting stabilized + 10x augmented Pix2Pix training on {device}...")
    
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

            # Target output sizes are 30x30 for standard PatchGAN
            valid = torch.ones((real_A.size(0), 1, 30, 30), device=device)
            fake = torch.zeros((real_A.size(0), 1, 30, 30), device=device)
            
            # Label smoothing + Label noise
            valid_smooth = torch.full((real_A.size(0), 1, 30, 30), 0.9, device=device) + torch.randn((real_A.size(0), 1, 30, 30), device=device) * 0.02
            fake_smooth = torch.full((real_A.size(0), 1, 30, 30), 0.1, device=device) + torch.randn((real_A.size(0), 1, 30, 30), device=device) * 0.02
            
            valid_smooth = torch.clamp(valid_smooth, 0.0, 1.0)
            fake_smooth = torch.clamp(fake_smooth, 0.0, 1.0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Generate fake image
            fake_B = generator(real_A)
            
            # Concatenate real target face and detached fake face along batch dimension
            real_fake_B = torch.cat((real_B, fake_B.detach()), dim=0)
            real_A_double = torch.cat((real_A, real_A), dim=0)
            
            # Single forward pass through Discriminator
            pred_all = discriminator(real_A_double, real_fake_B)
            
            # Split the predictions back into real and fake
            pred_real, pred_fake = torch.split(pred_all, real_A.size(0), dim=0)
            
            # Calculate loss with soft/noisy labels
            loss_real = criterion_GAN(pred_real, valid_smooth)
            loss_fake = criterion_GAN(pred_fake, fake_smooth)
            
            loss_D = 0.5 * (loss_real + loss_fake)
            accelerator.backward(loss_D)
            optimizer_D.step()

            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            
            # Evaluate fake image through updated Discriminator
            pred_fake = discriminator(real_A, fake_B)
            
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            
            # Total G Loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            accelerator.backward(loss_G)
            optimizer_G.step()

            gathered_loss_G = accelerator.gather_for_metrics(loss_G).mean().item()
            gathered_loss_D = accelerator.gather_for_metrics(loss_D).mean().item()

            epoch_loss_G += gathered_loss_G
            epoch_loss_D += gathered_loss_D

            if accelerator.is_main_process:
                pbar.set_postfix({"loss_G": gathered_loss_G, "loss_D": gathered_loss_D})
            
        if accelerator.is_main_process:
            history_loss_G.append(epoch_loss_G / len(dataloader))
            history_loss_D.append(epoch_loss_D / len(dataloader))

            with open("loss_history.json", "w") as f:
                json.dump({"loss_G": history_loss_G, "loss_D": history_loss_D}, f)
                
            if (epoch + 1) % 50 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                
                unwrapped_gen = accelerator.unwrap_model(generator)
                unwrapped_disc = accelerator.unwrap_model(discriminator)
                
                torch.save(unwrapped_gen.state_dict(), f"checkpoints/G_{epoch+1}.pth")
                torch.save(unwrapped_disc.state_dict(), f"checkpoints/D_{epoch+1}.pth")
                
                prev_epoch = (epoch + 1) - 50
                if prev_epoch > 0:
                    for prefix in ["G", "D"]:
                        old_ckpt = f"checkpoints/{prefix}_{prev_epoch}.pth"
                        if os.path.exists(old_ckpt):
                            os.remove(old_ckpt)

if __name__ == '__main__':
    train()
