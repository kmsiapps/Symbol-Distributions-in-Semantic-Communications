import shutil

source_dir = '/datasets/ImageNet/'
dest_dir = '/datasets/ImageNet8000/'
image_list_file = 'utils/ImageNetLarge8000.txt'

with open(image_list_file, 'r') as f:
    image_list = f.readlines()

for i, image in enumerate(image_list):
    image_fn = source_dir + image.strip()
    shutil.copy(image_fn, dest_dir+f'img_{i+1}.jpg')