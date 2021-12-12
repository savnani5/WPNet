import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, model, trainloader, epochs, device, lr=0.001 , weight_decay=1e-5):
        self.criterion = nn.MSELoss()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.trainloader = trainloader 
        self.device = device


    def loss_function(self, image_batch, waypoint_batch, mode='train'):  
        # zero the parameter gradients
        self.optimizer.zero_grad()

        #forward + backward + optimize
        outputs = self.model(image_batch)
        loss = self.criterion(outputs, waypoint_batch)
        if mode == 'train':
            loss.backward()
            self.optimizer.step()

        return loss


    def train(self):
        
        print("__________________Training Started__________________")
        with open(f"loss_files/loss_v1_manual_0.txt", "w") as f:
            for epoch in range(self.epochs):  # loop over the dataset multiple times
                
                running_loss = 0.0
                for i, sample_batch in enumerate(tqdm(self.trainloader)):
                    image_batch = sample_batch['image'].to(self.device).float()
                    waypoint_batch = sample_batch['waypoint'].to(self.device).float()

                    # Calculating loss
                    loss = self.loss_function(image_batch, waypoint_batch, mode='train')

                    # print statistics
                    running_loss += float(loss.item())
                    if i % 20 == 19:    # print every 20 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / 20))
                        f.write(f"{epoch+1}\t{i+1}\t{running_loss / 20}\n")
                        running_loss = 0.0
            
            # Saving intermediate models after each epoch
            if epoch > 5 and epoch % 2 == 0:
                torch.save(self.model.state_dict(),  f'/final_models/wpnet_{epoch}.pt')
            
        print('___________________Finished Training___________________')
        return self.model

    # def test(self, dataloader):
    #     # You cannot plot in 7 dimensions so make 7 plots !
        
    #     # Define figure for plots 
    #     fig, ax = plt.subplots(3,4)
    #     # Plots
    #     for i, sample_batch in enumerate(dataloader):
    #         image_batch, waypoint_batch = sample_batch['image'], sample_batch['waypoint']
            
    #         # Calculating loss
    #         # loss = self.loss_function(image_batch, waypoint_batch, mode='test')
    #         output_batch = self.model(image_batch.float()).detach().numpy()
    #         waypoint_batch = waypoint_batch.numpy()

    #         for i in range(3):
    #             ax[0, i].scatter(output_batch[:,i], waypoint_batch[:,i])
    #             plt.plot(X_train.detach().numpy()[:100] , predicted[:100] , "red")

    #     plt.xlabel("Celcius")
    #     plt.ylabel("Farenhite")
    #     plt.show()
