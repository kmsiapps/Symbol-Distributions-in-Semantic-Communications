
from PIL import Image
import torch
import torchvision
from utils.dataset import indexedImageDataset

__all__ = ['loadYooHanJu', 'loadCroppedKodak']

def loadYooHanJu():
    image = Image.open('images/sample_yhj.jpg').convert("RGB")
    image = torchvision.transforms.ToTensor()(image)
    image = image.unsqueeze(0).to('cuda')
    
    return image

def loadCroppedKodak():
    patch_size = (256, 256)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(patch_size),
        torchvision.transforms.ToTensor()
    ])
    dataset = indexedImageDataset(
        root_dir='/datasets/Kodak/',
        fname_format="kodim{:02}.png",
        num_images=24,
        transform=transform
    )
    images = []
    for i in range(24):
        images.append(dataset[i].unsqueeze(0))
    images = torch.concat(images, dim=0).to('cuda')
    
    return images
    
    