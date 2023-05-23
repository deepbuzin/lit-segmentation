import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
from glob import glob

data_root = '../notebooks/data/leaf_disease/data'


def fetch_pair(image_path, mask_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (512, 512))
    mask = cv2.resize(mask, (512, 512))

    blend = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    merge = np.concatenate((img, mask, blend), axis=1, dtype=np.uint8)

    return merge


def get_file_list(root):
    image_list = sorted(glob(os.path.join(root, 'images', '*.jpg')))
    mask_list = sorted(glob(os.path.join(root, 'masks', '*.png')))
    assert len(image_list) == len(mask_list)

    pairs = [(image_path, mask_path) for image_path, mask_path in zip(image_list, mask_list)]
    return pairs



if __name__ == '__main__':
    # np.random.seed(1337)

    pairs = get_file_list(data_root)

    random_indices = np.random.randint(0, len(pairs), 10)
    random_pairs = [pairs[i] for i in random_indices]

    vis_img = np.concatenate([fetch_pair(*pair) for pair in random_pairs], axis=0)
    plt.imshow(vis_img)
    plt.show()

    # random_pair = pairs[np.random.randint(0, len(pairs))]
    # plt.imshow(fetch_pair(*random_pair))
    # plt.show()


