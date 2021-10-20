import torch
import torch.nn as nn
from torchsummary import summary



##################################################
#       RESNET
################################################


# Adapted from:
# https://github.com/Raschka-research-group/coral-cnn/blob/master/model-code/afad-ce.py

def conv3x3kernel(in_channels, out_channels, stride=1):
    """2D convolution layer with kernel_size=(3x3)"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """Single residual_block of Resnet"""
    
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3kernel(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3kernel(out_channels, out_channels)
        self.bn2 =nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        # storing residual to feed to
        # the last layer
        residual = x
        
        # first sub_block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # second subbolock
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        # adding residual stored at 
        # the start to the output from conv layer
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.in_dim = 3
        # first conv layer in resnet
        self.conv1 = nn.Conv2d(self.in_dim, out_channels=64, kernel_size=7,\
                               stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 1)
        

        #Initializing weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    # generating layers
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        #Flattening the layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




#############################################
#       NIU CNN
#########################################

class Niu_model(nn.Module):

    def __init__(self):
        super(Niu_model, self).__init__()
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=20,
                          kernel_size=(5, 5),
                          stride=(1, 1)),
                nn.ReLU(),
                nn.LayerNorm([20, 56, 56]),
                nn.MaxPool2d(kernel_size=(2, 2)))
        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=20,
                          out_channels=40,
                          kernel_size=(7, 7),
                          stride=(1, 1)),
                nn.ReLU(),
                nn.LayerNorm([40,22,22]),
                nn.MaxPool2d(kernel_size=(2, 2)))

        self.block_3  = nn.Sequential(
                nn.Conv2d(in_channels=40,
                          out_channels=80,
                          kernel_size=(11, 11),
                          stride=(1, 1)),
                nn.ReLU(),
                nn.LayerNorm([80, 1, 1]))
        self.regressor = nn.Sequential(
                    nn.Linear(80*1*1, 1))


    def forward(self, x):

        x = self.block_1(x)
        #print(x.shape)
        x = self.block_2(x)
        #print(x.shape)
        x = self.block_3(x)
        #print(x.shape)
        x = self.regressor(x.view(x.size(0), -1))
        return x


####################################################


#############################################
#       NIU  forward backward
#########################################

class Niu_forback(nn.Module):

    def __init__(self):
        super(Niu_forback, self).__init__()
        self.block_x1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=20,
                          kernel_size=(5, 5),
                          stride=(1, 1)),
                nn.ReLU(),
                nn.LayerNorm([20, 56, 56]),
                nn.MaxPool2d(kernel_size=(2, 2)))
        self.block_x2 = nn.Sequential(
                nn.Conv2d(in_channels=20,
                          out_channels=40,
                          kernel_size=(7, 7),
                          stride=(1, 1)),
                nn.ReLU(),
                nn.LayerNorm([40,22,22]),
                nn.MaxPool2d(kernel_size=(2, 2)))

        self.block_x3  = nn.Sequential(
                nn.Conv2d(in_channels=40,
                          out_channels=80,
                          kernel_size=(11, 11),
                          stride=(1, 1)),
                nn.ReLU(),
                nn.LayerNorm([80, 1, 1]))
        self.regressor = nn.Sequential(
                    nn.Linear(80*1*1, 1))
        self.block_s1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=20,
                          kernel_size=(11, 11),
                          stride=(1, 1)),
                nn.Tanh(),
                nn.LayerNorm([20, 50, 50]),
                nn.AvgPool2d(kernel_size=(2, 2)))

        self.block_s2 = nn.Sequential(
                nn.Conv2d(in_channels=20,
                          out_channels=40,
                          kernel_size=(7, 7),
                          stride=(1, 1)),
                nn.Tanh(),
                nn.LayerNorm([40,19,19]),
                nn.AvgPool2d(kernel_size=(2, 2)))

        self.block_s3  = nn.Sequential(
                nn.Conv2d(in_channels=40,
                          out_channels=80,
                          kernel_size=(5, 5),
                          stride=(1, 1)),
                nn.Tanh(),
                nn.LayerNorm([80, 5, 5]))
        self.block_s4 = nn.Sequential(
                nn.Conv2d(in_channels=80,
                          out_channels=80,
                          kernel_size=(5, 5),
                          stride=(1, 1)),
                nn.Tanh(),
                nn.LayerNorm([80, 1, 1]))


    def forward(self, input):

        x = self.block_x1(input)
        s = self.block_s1(input)
        #print(x.shape)
        x = self.block_x2(x)
        s = self.block_s2(s)
        #print(x.shape)
        x = self.block_x3(x)
        s = self.block_s3(s)
        #print(x.shape)
        s = self.block_s4(s)
        o = (x + s)/2
        o = self.regressor(o.view(o.size(0), -1))
        return o



class Niu_modified(nn.Module):

    def __init__(self):
        super(Niu_modified, self).__init__()
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=32,
                          kernel_size=(5, 5),
                          stride=(1, 1)),
                nn.ReLU(),
                nn.LayerNorm([32, 56, 56]),
                nn.MaxPool2d(kernel_size=(2, 2)))
        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=(5, 5),
                          stride=(1, 1)),
                nn.ReLU(),
                nn.LayerNorm([64,24,24]),
                nn.MaxPool2d(kernel_size=(2, 2)))

        self.block_3  = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=(5,5),
                          stride=(1, 1)),
                nn.ReLU(),
                nn.LayerNorm([128, 8, 8]),
                nn.MaxPool2d(kernel_size=(2, 2)))
        self.block_4  = nn.Sequential(
                nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=(4,4),
                          stride=(1, 1)),
                nn.ReLU(),
                nn.LayerNorm([256, 1, 1]))
        self.regressor = nn.Sequential(
                    nn.Linear(256*1*1, 1))


    def forward(self, x):

        x = self.block_1(x)
        #print(x.shape)
        x = self.block_2(x)
        #print(x.shape)
        x = self.block_3(x)
        #print(x.shape)
        x = self.block_4(x)
        x = self.regressor(x.view(x.size(0), -1))
        return x










####################################################


def resnet18():
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2])
    return model


def niu_model():
    return Niu_model()



def niu_forback():
	return Niu_forback()


def niu_modified():
    return Niu_modified()


if __name__=="__main__":
    model = resnet18()
    print(model)
    print(summary(model, input_size=(3, 120, 120)))

