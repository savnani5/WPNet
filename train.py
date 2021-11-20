import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from model import WPNet, PTModel
from dataloader import SastaDataset, Rescale, ToTensor


def loss_function(model, y_true, y_pred, mode='train'):
    model.opt.zero_grad()
    # What to use ?? - Decide later
    criterion = nn.MSELoss()
    loss = criterion(y_true, y_pred)
    if mode == 'train':
        loss.backward()
        model.opt.step()
    return loss


def train(model, trainloader, epochs):
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        
        for i_batch, sample_batched in enumerate(trainloader):
            image_batch = sample_batched['image']
            waypoint_batch = sample_batched['waypoint']

            # zero the parameter gradients
            model.opt.zero_grad()

            # forward + backward + optimize
            outputs = model(image_batch)
            loss = loss_function(model, waypoint_batch, outputs)

            # print statistics
            running_loss += loss.item()
            if i_batch % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i_batch + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return model



def main():

    # Varibale initialization
    save_model_path = 'final_models/'
    txt_file = "../Dataset/manual_1/manual_1/airsim_rec.txt"
    img_dir = "../Dataset/manual_1/manual_1/images"
    batch_size = 4
    epochs = 2
    desired_image_input = (3, 432, 768)

    # Generating the dataloader
    dataset = SastaDataset(txt_file=txt_file, img_dir=img_dir, transform=transforms.Compose([Rescale(desired_image_input[1], desired_image_input[2]),ToTensor()]))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Loading the pretrained monocular depth estimation model 
    tensor = torch.zeros((batch_size, desired_image_input[0], desired_image_input[1], desired_image_input[2]))
    model = PTModel().float()
    model.load_state_dict(torch.load("models/nyu.pt"))
    wpnet = WPNet(tuple(tensor.shape), model)

    # Model training
    trained_model = train(wpnet, trainloader, epochs=epochs)
    
    # Not saving the full model - only the state dictionary - might modify to add the current state of model - like epochs trained, batch size etc.
    torch.save(trained_model.state_dict(), save_model_path)

    

if __name__ == "__main__":
    main()