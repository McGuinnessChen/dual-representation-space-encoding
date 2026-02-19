import tensorflow_datasets as tfds
import os

# 设置本地缓存路径
data_dir = ('./tensorflow_datasets')

# 下载 Omniglot 数据集并保存到本地
print("Downloading Omniglot...")
tfds.load(
    'omniglot',
    split='train',
    as_supervised=True,
    shuffle_files=False,
    try_gcs=False,
    data_dir=data_dir
)
print("Download complete. Data stored in:", data_dir)