from model import Generator,Discriminator
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,InterpolationMode
import os
from PIL import Image
from train import transfer_model
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    Crop = Compose([
            RandomCrop(256),ToTensor(),
        ])

    img_dir = '/Users/wangyanli/Downloads/SmallData/all'
    img_list = [img_dir+'/'+i for i in os.listdir(img_dir)]

    netG = Generator(scale_factor=4)
    netG = transfer_model('./model_data/netG_epoch_5_30.042746_0.776071.pth',netG)
    for img in img_list[:1]:
        plt.figure()
        plt.subplot(131)
        image = Image.open(img)
        plt.imshow(np.array(image))
        plt.subplot(132)
        image = Crop(image).unsqueeze(0)
        plt.imshow(image.squeeze(0).permute(1,2,0).numpy())
        plt.subplot(133)
        image = netG(image)
        rs_image = image.detach().permute(0,2,3,1).squeeze(0).numpy()
        plt.imshow(rs_image)
        plt.savefig('./1.jpg')




