from torch.utils.data.dataset import Dataset
from PIL import Image
from os.path import join
from torchvision.transforms import Compose,GaussianBlur, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,InterpolationMode
from os import listdir

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class TrainDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = crop_size//upscale_factor*upscale_factor
        self.hr_transform = Compose([
            RandomCrop(crop_size),ToTensor(),
        ])
        self.lr_transform = Compose([
            GaussianBlur(kernel_size=5),
            ToPILImage(),Resize(crop_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),ToTensor()
        ])

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class ValDataset(Dataset):
    def __init__(self, dataset_dir, crop_size,upscale_factor):
        super(ValDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.crop_size = crop_size//upscale_factor*upscale_factor
        self.lr_scale = Resize(self.crop_size // self.upscale_factor, interpolation=InterpolationMode.BICUBIC)
        # self.hr_scale = Resize(self.crop_size, interpolation=InterpolationMode.BICUBIC)
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        hr_image = CenterCrop(self.crop_size)(hr_image)

        lr_image = self.lr_scale(hr_image)
        # hr_restore_img = self.hr_scale(lr_image)
        return ToTensor()(lr_image),ToTensor()(hr_image) # ToTensor()(hr_restore_img)

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    data_train = TrainDataset(r'F:\part\xqs\models\xqs\segformer-pytorch-master_c_2\VOCdevkit\VOC2007\JPEGImages',512, 8)
    for i,(lr,hr) in enumerate(data_train):
        plt.figure()
        plt.subplot(121)
        plt.imshow(np.transpose(lr,[1,2,0]))
        plt.subplot(122)
        plt.imshow(np.transpose(hr,[1,2,0]))
        plt.show()
        # print(lr.shape,hr.shape)
        # break