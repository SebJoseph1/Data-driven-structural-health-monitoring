from torch import nn
import torch

from nets.vgg import VGG16


class block(nn.Module):
    def __init__(self, in_size, out_size):
        super(block, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Model(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Model, self).__init__()
        self.vgg = VGG16(pretrained=pretrained)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = block(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = block(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = block(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = block(in_filters[0], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)

        return final

class UseModel():
    def __init__(self,weight=None):
        self.model = Model()
        if weight is not None:
            self.model.load_state_dict(torch.load(weight))
    
    def predict(self,x):
        x = torch.unsqueeze(x, dim=0)
        with torch.no_grad():
            y = self.model(x)
            y = torch.squeeze(y,dim=0)
            y = torch.argmax(y,dim=0)
            y = torch.where(y == 1, torch.tensor(255), y)
            return y
