import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt
import os

IMG_SIZE = 64
PATCH_SIZE = 16
MAX_COORD = IMG_SIZE - PATCH_SIZE

BATCH_SIZE = 64 


class ImagePatchDataset(Dataset):
    """
    A custom PyTorch Dataset that generates patch localization samples from a source dataset.
    
    For each image, it:
    1. Resizes and converts the image to grayscale.
    2. Randomly crops a `patch` of a specified size.
    3. Adds a small amount of Gaussian noise to the patch.
    4. Returns the source image, the noisy patch, and the ground truth coordinates.
    """
    def __init__(self, source_dataset, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
                 deterministic: bool = False, seed: int = 42):
        self.dataset = source_dataset
        self.patch_size = patch_size
        self.img_size = img_size
        self.deterministic = deterministic
        self.seed = seed
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.Grayscale(),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source_pil, _ = self.dataset[idx]
        source_image = self.transform(source_pil)

        if self.deterministic:
            rng = np.random.RandomState(self.seed + idx)
        else:
            rng = np.random

        max_coord = self.img_size - self.patch_size
        y_start = rng.randint(0, max_coord + 1)
        x_start = rng.randint(0, max_coord + 1)
        
        clean_patch = source_image[:, y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
        
        noise = torch.randn_like(clean_patch) * 0.01
        final_patch = torch.clamp(clean_patch + noise, 0, 1)

        return {
            'source_image': source_image,
            'patch': final_patch,
            'ground_truth_coords': torch.tensor([y_start, x_start], dtype=torch.long)
        }


def run_candidate_workflow():
    """
    Your workflow demonstreation.
    Loads the provided data indices, creates the dataset, and visualizes a sample.
    """

    if not os.path.exists('train_val_indices.pt'):
        print("Error: 'train_val_indices.pt' not found.")
        print("Please run the one-time data preparation step first.")
        return

    # 1. Load the source dataset (will use cached version if already downloaded -> download=False)
    download_data = True
    print("Loading source dataset (CIFAR-100)...")
    source_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=download_data)

    # 2. Custom dataset wrapper
    full_custom_dataset = ImagePatchDataset(source_dataset)

    # 3. Load the provided indices and create the dataset subset
    print("Loading provided indices and creating the final dataset...")
    
    candidate_indices = torch.load('train_val_indices.pt', weights_only=False)
    
    candidate_dataset = Subset(full_custom_dataset, candidate_indices)
    print(f"Dataset ready with {len(candidate_dataset)} samples.")

    # 4. Create a DataLoader 
    # Create own train/validation splits from this dataset. (free to do as you wish)
    data_loader = DataLoader(candidate_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Visualize random item from the dataset to verify
    print("\nVisualizing random sample from the dataset...")
    
    sample_batch = next(iter(data_loader))
    
    source_img = sample_batch['source_image'][0].squeeze().numpy()
    patch_img = sample_batch['patch'][0].squeeze().numpy()
    coords = sample_batch['ground_truth_coords'][0].numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(source_img, cmap='gray')
    rect = plt.Rectangle((coords[1], coords[0]), PATCH_SIZE, PATCH_SIZE, edgecolor='r', facecolor='none', lw=2)
    axs[0].add_patch(rect)
    axs[0].set_title(f"Source Image (GT at {coords})")
    axs[0].axis('off')

    axs[1].imshow(patch_img, cmap='gray')
    axs[1].set_title("Noisy Patch")
    axs[1].axis('off')
    
    plt.suptitle("Data Generation Example")
    plt.show()


if __name__ == '__main__':
    run_candidate_workflow()