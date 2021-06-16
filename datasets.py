
from torch.utils.data import Dataset

from PIL import Image
import os

from scipy import misc


def load_img(filepath):
    img = misc.imread(filepath)
    img = Image.fromarray(img)
    return img

class BaseDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
       
        imgs = os.listdir(self.image_dir)
        self.img_paths = []
        self.img_names = []
        for img in imgs:
            cur_path = os.path.join(self.image_dir, img)
            self.img_paths.append(cur_path)
            self.img_names.append(img.split('.')[0])
        self.num = len(self.img_paths)
        self.img_size = len(self.img_paths)
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        cur_path = self.img_paths[index % self.img_size]
        cur_img = load_img(cur_path)
        if self.transform is not None:
            cur_img = self.transform(cur_img)

               
        cur_name = self.img_names[index % self.img_size]
        return {'img': cur_img, 'path': cur_path, 'name': cur_name }
