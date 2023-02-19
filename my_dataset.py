import os

import PIL
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(MyDataset, self).__init__()
        data_root = os.path.join(root, "kneedata", "training" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.flag = "training" if train else "test"
        self.transforms = transforms

        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".png")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        mask_names = [i for i in os.listdir(os.path.join(data_root, "masks")) if i.endswith(".png")]
        self.mask_list = [os.path.join(data_root, "masks", i) for i in mask_names]

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = img.resize((256, 256))
        #img = img.resize((512, 512))
        img = np.array(img)
        mask = Image.open(self.mask_list[idx]).convert('L')
        mask = mask.resize((256, 256))
        # 阈值处理掩模图
        mask = np.array(mask)/ 255
        # mask[mask == 64] = 0
        # mask[mask == 191] = 0
        # mask[mask == 128] = 255
        #mask = mask.resize((512, 512))
        # mask = mask / 255

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)
        img = Image.fromarray(img)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

