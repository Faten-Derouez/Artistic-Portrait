"""
Artistic Portrait Optimization using Colour Migration and GAN Enhancement
Based on: "Colour Migration and Generative Adversarial Network Based Enhancement
Techniques for Artistic Portrait Optimization"

Author: Implementation based on the paper methodology
Date: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.linalg import sqrtm
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# PART 1: DATA COLLECTION & PRE-PROCESSING
# ============================================================

class PortraitDataset(Dataset):
    """
    Dataset class for loading portrait paintings
    Dataset source: https://www.kaggle.com/datasets/deewakarchakraborty/portrait-paintings
    """

    def __init__(self, data_dir, image_size=(128, 128), augment=True):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.glob('*.jpg')) + list(self.data_dir.glob('*.png'))
        self.image_size = image_size
        self.augment = augment

        # Base transform: Resize and normalize
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        # Augmentation transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.augment:
            image = self.augment_transform(image)

        image = self.base_transform(image)
        return image


def rgb_to_hsv_tensor(image):
    """
    Convert RGB tensor [0,1] to HSV color space
    Based on equation (2) and (3) from the paper
    """
    r, g, b = image[0], image[1], image[2]

    max_val = torch.max(image, dim=0)[0]
    min_val = torch.min(image, dim=0)[0]
    delta = max_val - min_val

    # Value channel (V)
    v = max_val

    # Saturation channel (S) - equation (2)
    s = torch.where(max_val == 0, torch.zeros_like(max_val), delta / max_val)

    # Hue channel (H) - equation (3)
    h = torch.zeros_like(max_val)
    mask_r = (max_val == r) & (delta != 0)
    mask_g = (max_val == g) & (delta != 0)
    mask_b = (max_val == b) & (delta != 0)

    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    h = h / 360.0  # Normalize to [0, 1]

    return torch.stack([h, s, v])


# ============================================================
# PART 2: COLOUR MIGRATION (Based on Reinhard et al. 2001)
# ============================================================

class ColourMigration:
    """
    Colour Migration module for transferring colour distributions
    Implements the Mean-Variance Colour Transfer in LAB space
    Based on equation (5), (6), (7) from the paper
    """

    @staticmethod
    def rgb_to_lab(image):
        """
        Convert RGB image to LAB color space
        LAB separates luminance (L) from color (a, b)
        """
        if isinstance(image, torch.Tensor):
            image = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        elif image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return lab.astype(np.float32)

    @staticmethod
    def lab_to_rgb(lab):
        """Convert LAB image back to RGB"""
        lab = lab.astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return rgb

    @staticmethod
    def color_transfer_mean_std(target, reference):
        """
        Statistical colour transfer using mean and standard deviation
        Implements equation (5): C_t' = (σ_r/σ_t)(C_t - μ_t) + μ_r

        Parameters:
        - target: Target portrait image (RGB)
        - reference: Reference image with desired colour palette

        Returns:
        - Colour-migrated portrait
        """
        # Convert both images to LAB space
        target_lab = ColourMigration.rgb_to_lab(target)
        ref_lab = ColourMigration.rgb_to_lab(reference)

        # Compute mean and standard deviation for each channel - equation (5)
        target_mean = target_lab.mean(axis=(0, 1))
        target_std = target_lab.std(axis=(0, 1))
        ref_mean = ref_lab.mean(axis=(0, 1))
        ref_std = ref_lab.std(axis=(0, 1))

        # Apply colour transfer formula
        transferred_lab = np.zeros_like(target_lab)
        for channel in range(3):
            transferred_lab[:, :, channel] = (target_lab[:, :, channel] - target_mean[channel]) * \
                                             (ref_std[channel] / (target_std[channel] + 1e-8)) + \
                                             ref_mean[channel]

        # Clip values to valid range
        transferred_lab = np.clip(transferred_lab, 0, 255)

        # Convert back to RGB
        transferred_rgb = ColourMigration.lab_to_rgb(transferred_lab)

        return transferred_rgb

    @staticmethod
    def covariance_matching_transfer(target, reference):
        """
        Advanced colour transfer using mean and covariance matching
        Implements equations (6) and (7): x_t' = A(x_t - μ_t) + μ_r
        where A = Σ_r^(1/2) Σ_t^(-1/2)
        """
        # Reshape images to (N_pixels, 3)
        target_pixels = target.reshape(-1, 3).astype(np.float64)
        ref_pixels = reference.reshape(-1, 3).astype(np.float64)

        # Compute means
        target_mean = target_pixels.mean(axis=0)
        ref_mean = ref_pixels.mean(axis=0)

        # Zero-center the data
        target_centered = target_pixels - target_mean

        # Compute covariance matrices
        target_cov = np.cov(target_centered.T)
        ref_cov = np.cov(ref_pixels.T)

        # Compute transformation matrix A = Σ_r^(1/2) Σ_t^(-1/2)
        # Using eigenvalue decomposition for matrix square root
        try:
            # Compute Σ_t^(-1/2)
            eigvals_t, eigvecs_t = np.linalg.eigh(target_cov)
            inv_sqrt_t = eigvecs_t @ np.diag(1.0 / np.sqrt(eigvals_t + 1e-8)) @ eigvecs_t.T

            # Compute Σ_r^(1/2)
            eigvals_r, eigvecs_r = np.linalg.eigh(ref_cov)
            sqrt_r = eigvecs_r @ np.diag(np.sqrt(eigvals_r + 1e-8)) @ eigvecs_r.T

            # Transform matrix A
            A = sqrt_r @ inv_sqrt_t

            # Apply transformation - equation (6)
            transformed_pixels = (target_centered @ A.T) + ref_mean

        except np.linalg.LinAlgError:
            # Fallback to mean-std transfer if covariance is singular
            print("Covariance matrix singular, falling back to mean-std transfer")
            return ColourMigration.color_transfer_mean_std(target, reference)

        # Clip and reshape
        transformed_pixels = np.clip(transformed_pixels, 0, 255).astype(np.uint8)
        transformed_image = transformed_pixels.reshape(target.shape)

        return transformed_image


# ============================================================
# PART 3: GENERATIVE ADVERSARIAL NETWORK (GAN)
# ============================================================

class Generator(nn.Module):
    """
    Generator network - creates enhanced artistic portraits
    Takes random noise and generates realistic/stylized images
    Uses transposed convolutions for upsampling
    """

    def __init__(self, latent_dim=100, img_channels=3, feature_map_size=64):
        super(Generator, self).__init__()

        self.initial_size = 4  # Start from 4x4 feature map

        # Fully connected layer to project latent vector to initial feature map
        self.fc = nn.Linear(latent_dim, feature_map_size * 8 * self.initial_size * self.initial_size)

        # Sequential upsampling blocks
        self.upsample_blocks = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            self._make_gen_block(feature_map_size * 8, feature_map_size * 8, first_block=True),
            # Block 2: 8x8 -> 16x16
            self._make_gen_block(feature_map_size * 8, feature_map_size * 4),
            # Block 3: 16x16 -> 32x32
            self._make_gen_block(feature_map_size * 4, feature_map_size * 2),
            # Block 4: 32x32 -> 64x64
            self._make_gen_block(feature_map_size * 2, feature_map_size),
            # Block 5: 64x64 -> 128x128
            self._make_gen_block(feature_map_size, feature_map_size),
        )

        # Final convolution to output RGB image
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(feature_map_size, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def _make_gen_block(self, in_channels, out_channels, first_block=False):
        if first_block:
            # First block: no batch norm after dense layer
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

    def forward(self, z):
        # z: random noise vector of shape (batch_size, latent_dim)
        x = self.fc(z)
        x = x.view(-1, 512, self.initial_size, self.initial_size)  # Reshape to feature map
        x = self.upsample_blocks(x)
        x = self.final_conv(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator network - distinguishes real from generated portraits
    Binary classifier using convolutional layers with LeakyReLU
    """

    def __init__(self, img_channels=3, feature_map_size=64):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            # Input: 128x128x3
            self._make_disc_block(img_channels, feature_map_size, kernel_size=4, stride=2, padding=1),
            # 64x64x64
            self._make_disc_block(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1),
            # 32x32x128
            self._make_disc_block(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1),
            # 16x16x256
            self._make_disc_block(feature_map_size * 4, feature_map_size * 8, kernel_size=4, stride=2, padding=1),
            # 8x8x512
            self._make_disc_block(feature_map_size * 8, feature_map_size * 16, kernel_size=4, stride=2, padding=1),
            # 4x4x1024
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_map_size * 16, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def _make_disc_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x.view(-1, 1)


class GANEnhancer:
    """
    GAN-based enhancer for artistic portrait optimization
    Implements adversarial training with generator and discriminator
    """

    def __init__(self, image_size=128, latent_dim=100, lr=0.0002, beta1=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.latent_dim = latent_dim

        # Initialize networks
        self.generator = Generator(latent_dim=latent_dim, img_channels=3).to(self.device)
        self.discriminator = Discriminator(img_channels=3).to(self.device)

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

    def _weights_init(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight, 0.0, 0.02)
        if isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)

    def train_epoch(self, dataloader, epochs=1, print_freq=100):
        """
        Train the GAN for one epoch

        Adversarial training process:
        1. Train Discriminator on real and fake images
        2. Train Generator to fool the Discriminator
        """
        for epoch in range(epochs):
            for i, real_images in enumerate(dataloader):
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)

                # Labels for real and fake
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # ========================================
                # Step 1: Train Discriminator
                # ========================================
                self.optimizer_D.zero_grad()

                # Real images
                real_outputs = self.discriminator(real_images)
                d_real_loss = self.criterion(real_outputs, real_labels)

                # Fake images from generator
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_images = self.generator(z)
                fake_outputs = self.discriminator(fake_images.detach())
                d_fake_loss = self.criterion(fake_outputs, fake_labels)

                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.optimizer_D.step()

                # ========================================
                # Step 2: Train Generator
                # ========================================
                self.optimizer_G.zero_grad()

                # Generate new fake images
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_images = self.generator(z)
                fake_outputs = self.discriminator(fake_images)

                # Generator tries to fool discriminator (make fake images classified as real)
                g_loss = self.criterion(fake_outputs, real_labels)
                g_loss.backward()
                self.optimizer_G.step()

                # Print progress
                if (i + 1) % print_freq == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], '
                          f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    def enhance(self, colour_migrated_image, num_refinements=50):
        """
        Enhance a colour-migrated portrait using the trained GAN

        This implements the post-GAN enhancement described in the paper
        """
        self.generator.eval()

        # Convert image to tensor
        if isinstance(colour_migrated_image, np.ndarray):
            if colour_migrated_image.max() <= 1.0:
                colour_migrated_image = (colour_migrated_image * 255).astype(np.uint8)
            image_tensor = transforms.ToTensor()(colour_migrated_image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            image_tensor = image_tensor * 2 - 1  # Scale to [-1, 1]

        # Find optimal latent vector for this image (encoder-free approach)
        z = torch.randn(1, self.latent_dim).to(self.device)
        z.requires_grad = True
        optimizer = optim.Adam([z], lr=0.01)

        for _ in range(num_refinements):
            optimizer.zero_grad()
            generated = self.generator(z)
            loss = nn.MSELoss()(generated, image_tensor)
            loss.backward()
            optimizer.step()

        # Generate enhanced image
        with torch.no_grad():
            enhanced = self.generator(z)

        # Convert back to numpy
        enhanced = (enhanced.squeeze().cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        return enhanced


# ============================================================
# PART 4: POST-PROCESSING (Sharpening and Denoising)
# ============================================================

class PostProcessor:
    """
    Post-processing module for final refinement
    Implements sharpening kernel (equation 9) and Gaussian smoothing (equation 10)
    """

    @staticmethod
    def apply_sharpening(image):
        """
        Apply sharpening filter to enhance edges and details
        Equation (9): Sharpening Kernel K = [0 -1 0; -1 5 -1; 0 -1 0]
        """
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)

        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    @staticmethod
    def apply_noise_reduction(image, kernel_size=3):
        """
        Apply Gaussian blur for noise reduction
        Equation (10): Gaussian kernel G = (1/16)[1 2 1; 2 4 2; 1 2 1]
        """
        gaussian_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=np.float32) / 16

        denoised = cv2.filter2D(image, -1, gaussian_kernel)
        return denoised

    @staticmethod
    def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        """
        Alternative: Bilateral filter for edge-preserving denoising
        """
        filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return filtered

    @staticmethod
    def enhance_final(image, apply_sharp=True, apply_denoise=True):
        """Complete post-processing pipeline"""
        result = image.copy()

        if apply_denoise:
            result = PostProcessor.apply_noise_reduction(result)

        if apply_sharp:
            result = PostProcessor.apply_sharpening(result)

        return result


# ============================================================
# PART 5: COMPLETE PIPELINE
# ============================================================

class ArtisticPortraitOptimizer:
    """
    Complete framework combining Colour Migration and GAN Enhancement
    Implements the full workflow described in Figure 1 of the paper
    """

    def __init__(self, image_size=128):
        self.image_size = image_size
        self.colour_migration = ColourMigration()
        self.gan_enhancer = GANEnhancer(image_size=image_size)
        self.post_processor = PostProcessor()

    def process_single_image(self, target_image, reference_image):
        """
        Process a single portrait through the complete pipeline

        Steps:
        1. Pre-processing (resize, normalize)
        2. Colour Migration
        3. GAN Enhancement
        4. Post-processing
        """
        # Step 1: Pre-processing
        target_resized = cv2.resize(target_image, (self.image_size, self.image_size))
        reference_resized = cv2.resize(reference_image, (self.image_size, self.image_size))

        # Step 2: Colour Migration - equation (5) from paper
        colour_migrated = self.colour_migration.color_transfer_mean_std(
            target_resized, reference_resized
        )

        # Step 3: GAN Enhancement
        # Note: GAN requires pre-training on dataset first
        # For single image processing, use the refine method
        gan_enhanced = self.gan_enhancer.enhance(colour_migrated)

        # Step 4: Post-processing with sharpening and denoising
        final_output = self.post_processor.enhance_final(gan_enhanced)

        return {
            'colour_migrated': colour_migrated,
            'gan_enhanced': gan_enhanced,
            'final': final_output
        }

    def train_gan(self, data_dir, batch_size=32, num_epochs=50):
        """Train the GAN component on portrait dataset"""
        dataset = PortraitDataset(data_dir, image_size=(self.image_size, self.image_size))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        print(f"Training GAN on {len(dataset)} images for {num_epochs} epochs...")
        self.gan_enhancer.train_epoch(dataloader, epochs=num_epochs)
        print("Training completed!")


# ============================================================
# PART 6: EVALUATION METRICS
# ============================================================

class EvaluationMetrics:
    """
    Quantitative evaluation metrics as described in paper Section 4.2
    - SSIM: Structural Similarity Index Measure
    - PSNR: Peak Signal-to-Noise Ratio
    - FID: Frechet Inception Distance
    """

    @staticmethod
    def compute_ssim(image1, image2):
        """Compute Structural Similarity Index (SSIM)"""
        if image1.max() <= 1.0:
            image1 = (image1 * 255).astype(np.uint8)
        if image2.max() <= 1.0:
            image2 = (image2 * 255).astype(np.uint8)

        return ssim(image1, image2, channel_axis=2, data_range=255)

    @staticmethod
    def compute_psnr(image1, image2):
        """Compute Peak Signal-to-Noise Ratio (PSNR)"""
        if image1.max() <= 1.0:
            image1 = (image1 * 255).astype(np.uint8)
        if image2.max() <= 1.0:
            image2 = (image2 * 255).astype(np.uint8)

        return psnr(image1, image2, data_range=255)

    @staticmethod
    def compute_histogram_similarity(image1, image2):
        """Compare colour histogram distributions"""
        hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity


# ============================================================
# PART 7: MAIN EXECUTION
# ============================================================

def main():
    """
    Complete execution of the Artistic Portrait Optimization framework
    """
    print("=" * 60)
    print("Artistic Portrait Optimization Framework")
    print("Colour Migration + GAN Enhancement")
    print("=" * 60)

    # Initialize the optimizer
    optimizer = ArtisticPortraitOptimizer(image_size=128)

    # ========================================
    # Step 1: Train GAN on portrait dataset
    # ========================================
    data_directory = "./portrait_dataset"  # Path to your dataset

    if os.path.exists(data_directory):
        print("\n[Step 1] Training GAN on portrait dataset...")
        # Uncomment to train: optimizer.train_gan(data_directory, batch_size=32, num_epochs=50)
        print("Training requires dataset. Loading pre-trained weights recommended.")

    # ========================================
    # Step 2: Load images for processing
    # ========================================
    print("\n[Step 2] Loading images...")

    # Example: Create sample images for demonstration
    # Replace with actual image loading in practice
    sample_target = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    sample_reference = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # ========================================
    # Step 3: Process through complete pipeline
    # ========================================
    print("\n[Step 3] Processing through Colour Migration...")
    print("[Step 4] Applying GAN Enhancement...")
    print("[Step 5] Post-processing with sharpening and denoising...")

    # Process the image
    results = optimizer.process_single_image(sample_target, sample_reference)

    print("\n[Step 6] Computing evaluation metrics...")

    # ========================================
    # Step 4: Evaluation
    # ========================================
    metrics = EvaluationMetrics()

    ssim_score = metrics.compute_ssim(results['final'], sample_target)
    psnr_score = metrics.compute_psnr(results['final'], sample_target)
    hist_similarity = metrics.compute_histogram_similarity(results['final'], sample_target)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"SSIM (Structural Similarity): {ssim_score:.4f}")
    print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr_score:.2f} dB")
    print(f"Histogram Similarity: {hist_similarity:.4f}")

    print("\n" + "=" * 40)
    print("FRAMEWORK EXECUTION COMPLETED")
    print("=" * 40)

    return results


if __name__ == "__main__":
    results = main()