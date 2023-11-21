import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        # vgg = vgg16(pretrained=True)
        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # for param in loss_network.parameters():
        #     param.requires_grad = False
        # self.loss_network = loss_network.cuda()
        self.local = local_feature_information()
        self.globe = global_feature_information()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        # perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        perception_loss = self.mse_loss(self.local(out_images), self.local(target_images)) + self.mse_loss(self.globe(out_images), self.globe(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class local_feature_information(nn.Module):
    def __init__(self):
        super(local_feature_information, self).__init__()
        from torchvision.models.resnet import resnet18
        resnet = resnet18(pretrained=True).eval()
        for param in resnet.parameters():
            param.requires_grad = False

        self.conv_bn_relu_maxpool = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = nn.Sequential(*list(resnet.layer1))
        self.layer2 = nn.Sequential(*list(resnet.layer2))
        self.layer3 = nn.Sequential(*list(resnet.layer3))
        self.layer4 = nn.Sequential(*list(resnet.layer4))

        del resnet
        torch.cuda.empty_cache()
        
    def minmax_scaler(self,x):
        normalized_tensor = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) 
        return normalized_tensor

    def forward(self,x):
        x = self.conv_bn_relu_maxpool(x)
        x1 = self.layer1(x)
        x2 =self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return self.minmax_scaler(x4)
    

class global_feature_information(nn.Module):
    def __init__(self):
        super(global_feature_information, self).__init__()
        from torchvision.models.swin_transformer import swin_t
        swin = swin_t(pretrained=True).eval()
        for param in swin.parameters():
            param.requires_grad = False

        self.layer1 = nn.Sequential(*list(swin.features)[:2],nn.Sigmoid())
        self.layer2 = nn.Sequential(*list(swin.features)[2:4],nn.Sigmoid())
        self.layer3 = nn.Sequential(*list(swin.features)[4:6],nn.Sigmoid())
        self.layer4 = nn.Sequential(*list(swin.features)[6:8],nn.Sigmoid())

        del swin
        torch.cuda.empty_cache()

    def forward(self,x):
        x1 = self.layer1(x)
        x2 =self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4.permute(0,3,1,2)

if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
