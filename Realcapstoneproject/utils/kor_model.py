import torch
import torch.nn as nn

from torchsummary import summary

class VGG16(nn.Module):

    def __init__(self, image_channels=3):
        super(VGG16, self).__init__()

        self.flatten = nn.Flatten()

        self.conv1 = nn.Sequential(
                nn.Conv2d(image_channels, 64, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2)
                )

        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2)
                )

        self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2)
                )

        self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2)
                )

        
        self.fully_connected_layer = nn.Sequential(
                nn.Linear(512 * 4 * 4, 4096),
                nn.BatchNorm1d(4096),
                nn.LeakyReLU(),
                nn.Dropout(0.5),

                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.LeakyReLU(),
                nn.Dropout(0.5),

                nn.Linear(4096, 45)
                )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fully_connected_layer(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    model = VGG16(1)
    model.to(device)
    summary(model, input_size=(1, 64, 64), batch_size=1, device=str(device))
