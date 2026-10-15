import torch
import torchvision
from torchvision.transforms.v2.functional import resize
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import os
import psutil
import io
from turbojpeg import TurboJPEG

class listedImageDataset(Dataset):
    def __init__(self, root_dir, image_list_file, cache, transform=None, mini=False):
        self.root_dir = root_dir
        with open(image_list_file, 'r') as f:
            self.image_list = f.readlines()
        self.transform = transform
        if mini:
            self.image_list = self.image_list[:100]
            
        if cache is not None: #use caching
            self.cache = cache
            self.RAM_crit = 21474836480 #20 GB
        else:
            self.cache = None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.cache is None:
            img_file = os.path.join(self.root_dir, self.image_list[idx].strip())
            image = Image.open(img_file)
            image = image.convert("RGB")
        else:
            if idx in self.cache.keys():
                image = self.cache[idx]
            else:    
                img_file = os.path.join(self.root_dir, self.image_list[idx].strip())
                image = Image.open(img_file)
                image = image.convert("RGB")
                self.cache[idx] = image
            
            available_RAM = psutil.virtual_memory().available
            if available_RAM < self.RAM_crit:
                pullout_idx = self.cache.keys()[0]
                del self.cache[pullout_idx]
        
        if self.transform:
            image = self.transform(image)
        return image
    
class indexedCachedImageDataset(Dataset):
    def __init__(self, root_dir, fname_format, num_images, transform=None):
        self.num_images = num_images
        self.transform = transform
        
        self.images = []
        for i in range(num_images):
            img_name = fname_format.format(i+1)
            img_file = os.path.join(root_dir, img_name)
            image = Image.open(img_file)
            image = image.convert("RGB")
            self.images.append(image)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
    
class indexedCodeCachedImageDataset(Dataset):
    def __init__(self, root_dir, fname_format, num_images, transform=None, mini=False):
        if mini:
            self.num_images = 100
        else:
            self.num_images = num_images
        self.transform = transform
        
        self.image_bytes = []
        for i in range(num_images):
            img_name = fname_format.format(i+1)
            img_file = os.path.join(root_dir, img_name)
            with open (img_file, 'rb') as f:
                img_bytes = f.read()
            self.image_bytes.append(img_bytes)
        
        self.jpeg = TurboJPEG()

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_bytes = self.image_bytes[idx]
        try:
            image = self.jpeg.decode(img_bytes)
            image = Image.fromarray(image)
            # Note:
            # apt install libturbojpeg
            # pip install pyturbojpeg
        except:
            image = Image.open(io.BytesIO(img_bytes))
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image
    
class indexedImageDataset(Dataset):
    def __init__(self, root_dir, fname_format, num_images, transform=None, mini=False):
        if mini:
            self.num_images = 64
        else:
            self.num_images = num_images
        self.transform = transform
        self.fname_format = fname_format
        self.root_dir = root_dir

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.fname_format.format(idx+1)
        img_file = os.path.join(self.root_dir, img_name)
        image = Image.open(img_file)
        image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image
    
class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img

class RandomResize(object):
    def __init__(self, random_range):
        if isinstance(random_range, int):
            self.random_min = random_range
            self.random_max = random_range + 1
        elif isinstance(random_range, tuple):
            assert isinstance(random_range[0], int)
            assert isinstance(random_range[1], int)
            self.random_min = random_range[0]
            self.random_max = random_range[1] + 1
        else:
            assert "random range should be int or tuple."

    def __call__(self, image):
        size = torch.randint(self.random_min, self.random_max, ())
        resized_image = resize(image, (size,))
        return resized_image

class addContinuousNoise(object):
    def __init__(self, image_size, clip_range, quantization_step):
        assert isinstance(image_size, tuple)
        assert isinstance(clip_range, tuple)
        assert isinstance(quantization_step, (int, float))
        self.image_size = image_size
        self.clip_range = clip_range
        self.quantization_step = quantization_step

    def __call__(self, image):
        noise = torch.rand(self.image_size) * self.quantization_step - (self.quantization_step/2)
        image.add_(noise)
        image.clamp_(min=self.clip_range[0], max=self.clip_range[1])
        return image


class RandomResizedCrop(object):
    def __init__(self, image_size_min, image_size_max, patch_size):
        self.image_size_min = image_size_min
        self.image_size_max = image_size_max
        self.patch_size = patch_size

    def __call__(self, image):
        h, w = image.size # PIL object
        image_size = max(h, w)
        min_ratio = self.image_size_min / image_size
        max_ratio = self.image_size_max / image_size
        tr =  torchvision.transforms.v2.RandomResizedCrop(
            size=self.patch_size, scale=(min_ratio, max_ratio)
        )
        image = tr(image)
        
        return image


def loadSingleDataset(type, root_dir, mini=False, cache=None):
    if type == 'ImageNet Large 8000':
        patch_size = (256, 256)
        transform = torchvision.transforms.v2.Compose([
            # RandomResize((640, 1200)),
            # RandomResizedCrop(
            #     640,
            #     1200,
            #     patch_size
            # ),
            torchvision.transforms.v2.RandomResize(640, 1200),
            torchvision.transforms.v2.RandomCrop(patch_size),
            torchvision.transforms.v2.ToImage(),
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            addContinuousNoise(
                image_size=patch_size,
                clip_range=(0, 1),
                quantization_step=1/256
            )
        ])
        dataset = indexedCodeCachedImageDataset(
            root_dir=root_dir,
            fname_format="img_{}.jpg",
            num_images=8000,
            transform=transform,
            mini=mini
        )
        # dataset = listedImageDataset(
        #     root_dir=root_dir,
        #     image_list_file='utils/ImageNetLarge8000.txt',
        #     transform=transform,
        #     cache=cache,
        #     mini=mini
        # )
    elif type == 'ImageNet Patch 80000':
        patch_size = (256, 256)
        transform = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.RandomCrop(patch_size),
            torchvision.transforms.v2.ToImage(),
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            addContinuousNoise(
                image_size=patch_size,
                clip_range=(0, 1),
                quantization_step=1/256
            )
        ])
        dataset = indexedImageDataset(
            root_dir=root_dir,
            fname_format="{}.jpg",
            num_images=80000,
            transform=transform,
            mini=mini
        )
    elif type == 'Kodak':
        patch_size = (256, 256)
        transform = torchvision.transforms.Compose([
            # RandomResize((640, 1200)),
            # torchvision.transforms.RandomCrop(patch_size),
            torchvision.transforms.ToTensor()
        ])
        dataset = indexedCachedImageDataset(
            root_dir=root_dir,
            fname_format="kodim{:02}.png",
            num_images=24,
            transform=transform
        )
    elif type == 'CIFAR10_train':
        transform = torchvision.transforms.ToTensor()
        raw_dataset = torchvision.datasets.CIFAR10(
            root=root_dir,
            train=True,
            transform=transform
        )
        dataset = DatasetWrapper(raw_dataset)
    elif type == 'CIFAR10_test':
        transform = torchvision.transforms.ToTensor()
        raw_dataset = torchvision.datasets.CIFAR10(
            root=root_dir,
            train=False,
            transform=transform
        )
        dataset = DatasetWrapper(raw_dataset)
        
    return dataset

    
def loadDatasets(
        train_data_type='ImageNet Large 8000',
        train_root_dir='/datasets/ImageNet',
        train_batch_size=8,
        valid_data_type='Kodak',
        valid_root_dir='/datasets/Kodak',
        num_cpus=8,
        shuffle_train=True,
        mini=False,
        pin_memory=False,
        distributed=False,
        cache=None
    ):
    
    train_dataset = loadSingleDataset(train_data_type, train_root_dir, mini=mini, cache=cache)
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=shuffle_train)
        batched_sampler = BatchSampler(train_sampler, train_batch_size, drop_last=False)
        train_data_loader = DataLoader(train_dataset,
                                       batch_sampler=batched_sampler,
                                       prefetch_factor=3,
                                       num_workers=num_cpus,
                                       pin_memory=pin_memory,
                                       persistent_workers=True
                                       )
    else:
        train_data_loader = DataLoader(train_dataset,
                                    batch_size=train_batch_size,
                                    shuffle=shuffle_train,
                                    num_workers=num_cpus,
                                    prefetch_factor=2,
                                    pin_memory=pin_memory,
                                    persistent_workers=True
                                    )
    valid_dataset = loadSingleDataset(valid_data_type, valid_root_dir)
    valid_data_loader = DataLoader(valid_dataset,
                                   shuffle=False,
                                   num_workers=num_cpus,
                                   pin_memory=pin_memory,
                                   persistent_workers=True)
    
    return train_data_loader, valid_data_loader
