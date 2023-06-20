import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm
import sys
sys.path.append('./utils')

from utils.dataloader_kor import Dataset_kor
from utils.kor_model import VGG16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    dataset = Dataset_kor(data_folder='./utils/json', transform=transforms.ToTensor())
    data_size = len(dataset)
    train_data, test_data = random_split(dataset, [int(0.8 * data_size), int(0.2 * data_size)])  

    train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=True, drop_last=False)

    # load model
    model = VGG16(image_channels=1)
    model.to(device)

    # define hyperparameters, loss function, optimizer
    learning_rate = 0.0001
    epochs = 60
    loss_fn = nn.CrossEntropyLoss().to(device)
    # loss_fn = nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
            lr_lambda=lambda epoch:0.95**epoch,
            last_epoch=-1,
            verbose=False)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n===============================================================")
        train(train_dataloader, model, optimizer, loss_fn, scheduler)
        test(test_dataloader, model, loss_fn)
        torch.save(model, 'total_test.pth')
    print("Done")


def train(dataloader, model, optimizer, loss_fn, scheduler):
    
    model.train()
    num_batches = len(dataloader)
    train_loss = 0

    for batch, (images, label) in enumerate(tqdm(dataloader)):
        images = images.to(device)  # (batch_size (N), 1, 64, 64)
        label = label.to(device)
        # one_hot =  torch.nn.functional.one_hot(label, num_classes =45).type(torch.float).to(device)
        
        predicted_label = model(images)

        # loss = loss_fn(predicted_label, one_hot)
        loss = loss_fn(predicted_label, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        # if batch % 300 == 0:
        #     loss, current = loss.item(), (batch+1)
        #     print(f"batch: {current}  loss: {loss:>7f}")
    
    train_loss /= num_batches
    print(f"loss: {train_loss}")
    scheduler.step()

def test(dataloader, model, loss_fn):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)  # (batch_size (N), 1, 64, 64)
            label = label.to(device)

            pred = model(images)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n\
            \Accuracy: {(100*correct):0.1f}%, Avg loss: {test_loss:>8f}\n")


if __name__ == '__main__':
    main()