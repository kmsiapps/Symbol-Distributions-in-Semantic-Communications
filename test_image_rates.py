#%%
from utils.dataset import loadDatasets
from torchvision.utils import save_image
import os
from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
train_dataloader, valid_dataloader = loadDatasets(
    train_data_type=  'ImageNet Large 8000',
    train_root_dir=   '/datasets/ImageNet/',
    train_batch_size= 1,
    valid_data_type=  'Kodak',
    valid_root_dir=   '/datasets/Kodak/',
    num_cpus=         1,
    mini=             False,
    pin_memory=       True,
    distributed=      False,
    cache=            None
)

file_sizes = []
temp_path = './temp/temp.png'
for img in tqdm(train_dataloader):
    save_image(img, temp_path)
    filesize = os.path.getsize(temp_path)
    file_sizes.append(filesize)

#%%

file_sizes_CIFAR = []
# Supported image file extensions
img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# Recursively walk through the directory
for root, _, files in tqdm(os.walk('/datasets/CIFAR10/')):
    for file in tqdm(files):
        if file.lower().endswith(img_extensions):
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_sizes_CIFAR.append(file_size)

file_sizes_CIFAR = np.array(file_sizes_CIFAR)
file_sizes = np.array(file_sizes)

#%%

file_sizes_CIFAR_norm = file_sizes_CIFAR / np.mean(file_sizes_CIFAR)
file_sizes_norm = file_sizes / np.mean(file_sizes)

plt.figure(figsize=(4, 2))
sns.histplot(file_sizes_CIFAR_norm)
sns.histplot(file_sizes_norm)
plt.legend(['CIFAR-10', 'ImageNet'])
plt.xlabel('Normalized Compressed Bytes')
plt.tight_layout()
plt.savefig('./image_rates.png')
plt.title('Compressed Byte Across Image Datasets')
plt.show()

np.savez_compressed('./imagenet_rates.npz', file_sizes)
np.savez_compressed('./cifar_rates.npz', file_sizes_CIFAR)

# %% Upper line from saved file

file_sizes_CIFAR = np.load('./results/raw_data/cifar_rates.npz')['arr_0']
file_sizes = np.load('./results/raw_data/imagenet_rates.npz')['arr_0']

file_sizes_CIFAR_norm = file_sizes_CIFAR / np.mean(file_sizes_CIFAR)
file_sizes_norm = file_sizes / np.mean(file_sizes)

plt.figure(figsize=(4, 2))
sns.histplot(file_sizes_CIFAR_norm)
sns.histplot(file_sizes_norm)
plt.legend(['CIFAR-10', 'ImageNet'])
plt.xlabel('Normalized Compressed Bytes')
plt.tight_layout()
plt.savefig('./images/image_rates.png')
# plt.title('Histogram of Compressed Byte Sizes Across Image Datasets')
plt.show()


