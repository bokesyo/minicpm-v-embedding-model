from pandas import read_parquet
from io import BytesIO
from PIL import Image

from matplotlib import pyplot as plt

# data = read_parquet("")
data = read_parquet('/Users/bokesyo/Downloads/arXiv_src_1811_002.parquet')
# pip install pyarrow

# locate image byte string
idx = 0
byte_string = data['caption_images'][idx][0]['cil_pairs'][0]['image']['bytes']
image = Image.open(BytesIO(byte_string))
image = image.convert("RGB")

# plt.imshow(Image.open(BytesIO(data['caption_images'][idx][0]['cil_pairs'][0]['image']['bytes'])).convert("RGB")); plt.show()
plt.imshow(image)
plt.show()

