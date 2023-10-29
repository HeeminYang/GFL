import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # ... [생략: 기존의 layer_x, layer_y, layer_xy 정의]

        # Adversarial output layer (real/fake) with Conv2d
        self.adv_layer = nn.Sequential(
            # Notice in below layer, we are using out channels as 1
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False),
            # out size : (batch_size, 1, 1, 1)
            nn.Sigmoid()
        )
        
        # Auxiliary output layer (class label prediction) with Conv2d
        self.aux_layer = nn.Sequential(
            # Notice in below layer, we are using out channels as 10 for 10 classes
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=3, stride=1, padding=0, bias=False),
            # out size : (batch_size, 10, 1, 1)
            nn.Softmax(dim=1)
        )

    def forward(self, img, labels):
        x = self.layer_x(img)
        # size of x : (batch_size, 32, 14, 14)
        y = self.layer_y(labels)
        # size of y : (batch_size, 32, 14, 14)
        xy = torch.cat([x,y], dim=1)
        # size of xy : (batch_size, 64, 14, 14)
        xy = self.layer_xy(xy)
        # size of xy : (batch_size, 256, 3, 3)
        
        validity = self.adv_layer(xy).view(xy.shape[0], -1)
        # size of validity : (batch_size, 1)
        label = self.aux_layer(xy).view(xy.shape[0], -1)
        # size of label : (batch_size, 10)

        return validity, label
  
class Generator(nn.Module):
  """ G(z) """
  def __init__(self, input_size=100):
    # initalize super module
    super(Generator, self).__init__()

    # noise z input layer : (batch_size, 100, 1, 1)
    self.layer_x = nn.Sequential(nn.ConvTranspose2d(in_channels=100, out_channels=128, kernel_size=3,
                                                  stride=1, padding=0, bias=False),
                                 # out size : (batch_size, 128, 3, 3)
                                 nn.BatchNorm2d(128),
                                 # out size : (batch_size, 128, 3, 3)
                                 nn.ReLU(),
                                 # out size : (batch_size, 128, 3, 3)
                                )

    # label input layer : (batch_size, 10, 1, 1)
    self.layer_y = nn.Sequential(nn.ConvTranspose2d(in_channels=10, out_channels=128, kernel_size=3,
                                                  stride=1, padding=0, bias=False),
                                 # out size : (batch_size, 128, 3, 3)
                                 nn.BatchNorm2d(128),
                                 # out size : (batch_size, 128, 3, 3)
                                 nn.ReLU(),
                                 # out size : (batch_size, 128, 3, 3)
                                )

    # noise z and label concat input layer : (batch_size, 256, 3, 3)
    self.layer_xy = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                                                  stride=2, padding=0, bias=False),
                               # out size : (batch_size, 128, 7, 7)
                               nn.BatchNorm2d(128),
                               # out size : (batch_size, 128, 7, 7)
                               nn.ReLU(),
                               # out size : (batch_size, 128, 7, 7)
                               nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                                                  stride=2, padding=1, bias=False),
                               # out size : (batch_size, 64, 14, 14)
                               nn.BatchNorm2d(64),
                               # out size : (batch_size, 64, 14, 14)
                               nn.ReLU(),
                               # out size : (batch_size, 64, 14, 14)
                               nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4,
                                                  stride=2, padding=1, bias=False),
                               # out size : (batch_size, 1, 28, 28)
                               nn.Tanh())
                               # out size : (batch_size, 1, 28, 28)

  def forward(self, x, y):
    # x size : (batch_size, 100)
    x = x.view(x.shape[0], x.shape[1], 1, 1)
    # x size : (batch_size, 100, 1, 1)
    x = self.layer_x(x)
    # x size : (batch_size, 128, 3, 3)

    # y size : (batch_size, 10)
    y = y.view(y.shape[0], y.shape[1], 1, 1)
    # y size : (batch_size, 100, 1, 1)
    y = self.layer_y(y)
    # y size : (batch_size, 128, 3, 3)

    # concat x and y
    xy = torch.cat([x,y], dim=1)
    # xy size : (batch_size, 256, 3, 3)
    xy = self.layer_xy(xy)
    # xy size : (batch_size, 1, 28, 28)
    return xy

