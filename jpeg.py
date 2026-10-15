#%% 
import numpy as np
import sys
import os
from PIL import Image
import io

fname_format="img_{}.jpg"
root_dir = '/datasets/ImageNet8000/'
idx = 1
img_name = fname_format.format(idx+1)
img_file = os.path.join(root_dir, img_name)

with open(img_file, 'rb') as f:
    bi = f.read()
print(bi)
# image = Image.open(img_file)
image = Image.open(io.BytesIO(bi))
# image = Image.frombytes(bi)
image = image.convert("RGB")
img = np.array(image)

print(sys.getsizeof(image))
print(sys.getsizeof(img))
print(1600*1200*3)
