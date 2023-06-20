import torch
import torch.nn as nn

from torchsummary import summary

class VGG16(nn.Module):

    def __init__(self, image_channels=3):
        super(VGG16, self).__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(image_channels, 64, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2)
                )

        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2)
                )

        self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2)
                )

        self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2)
                )

        self.conv5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2)
                )
        
        self.fcn = nn.Sequential(
                # after VGG: 512 x 8 x 8 (assume 1 x 256 x 256 input image)
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=0),
                nn.LeakyReLU(),


                nn.Conv2d(in_channels=1024, out_channels=4096, kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels=4096, out_channels=1024, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels=1024, out_channels=4, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
                )
    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fcn(x)

        return x

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Device : {device}")
#     model = VGG16(1)
#     model.to(device)
#     summary(model, input_size=(1, 256, 256), batch_size=1, device=str(device))