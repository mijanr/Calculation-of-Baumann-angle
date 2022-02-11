import torch
import torch.nn as nn
import torch.nn.functional as F



#create a pretrained resnet50 model
resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
#remove the last layer
resnet.fc = nn.Linear(resnet.fc.in_features, 8)
#create a CNN model object
class CNN(nn.Module):
    def __init__(self, output_dim:int)->None:
        '''
        parameters:
            output_dim: dimension of keypoints
        '''
        super(CNN, self).__init__()
        self.output_dim = output_dim
        self.model = nn.Sequential()
        
        #copy the pretrained resnet50 model
        #use one channel
        self.model.add_module('conv0', nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False))
        self.model.add_module('conv1', resnet.conv1)
        self.model.add_module('bn1', resnet.bn1)
        self.model.add_module('relu1', resnet.relu)
        self.model.add_module('maxpool1', resnet.maxpool)
        self.model.add_module('layer1', resnet.layer1)
        self.model.add_module('layer2', resnet.layer2)
        self.model.add_module('layer3', resnet.layer3)
        self.model.add_module('layer4', resnet.layer4)
        #flatten the output of the convolutional layer
        self.model.add_module('flatten',nn.Flatten())
        #add a fully connected layers
        self.model.add_module('fc1', nn.Linear(131072, 512))
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(512, 256))
        self.model.add_module('relu4', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(256, 128))
        self.model.add_module('relu5', nn.ReLU())  
        self.model.add_module('fc4', nn.Linear(128, 64))
        self.model.add_module('relu6', nn.ReLU())
        self.model.add_module('fc5', nn.Linear(64, output_dim))
    def forward(self, x: torch.Tensor)->torch.Tensor:
        '''
        parameters:
            x: input image in torch.Tensor format
        returns:
            output: output of the model in torch.Tensor format
        '''
        return self.model(x)
if __name__ == '__main__':
    #create an image of size 1x1x512x512
    img = torch.randn(1,1,512,512)
    #create a CNN model
    model = CNN(output_dim=8)
    #forward pass
    output = model(img)
    print(output.shape)
    