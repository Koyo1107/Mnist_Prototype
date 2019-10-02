import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
network = Net()
network.load_state_dict(torch.load('data/model.pth'))
network.eval()

def Image_tool(img):
    img = Image.open(img)
    img = img.resize((28,28))
    img = img.convert('L')
    
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
#        torchvision.transforms.Normalize(
#        mean=(0.1307,), std=(0.3081,))
    ])
    
#    to_tensor = transforms.ToTensor()
#    tensor = to_tensor(img)# pil image to torch.tensor
#    #tensor = tensor[None, :, :, :]
#    tensor = tensor.unsqueeze(0) # add more dimention in 0 dim
#    print(tensor.shape)
#    print(tensor.mean(), tensor.std(), tensor.min(), tensor.max())
#
#    tensor = 1 - tensor # invert color of tensor

 #   to_pil = transforms.ToPILImage()
 #   tensor_img = to_pil(tensor.squeeze(0))
 
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    #print('tensor:', tensor.shape, tensor.min(), tensor.mean(), tensor.max())

    out = network(tensor).data
    pred = out.squeeze()
    pred = pred.argmax()
    result = int(pred)
    return result

    
