# Import packages
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training and test dataset/dataloader
# Make a validation set with 10% of training data
cifar_train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.ToTensor())


imgs = torch.stack([i for i , _ in cifar_train_set],dim=3)
print(imgs.shape)
ds_mean = np.asarray(imgs.view(3,-1).mean(dim=1))
ds_std = np.asarray(imgs.view(3,-1).std(dim=1))
print(ds_mean)
print(ds_std)
# Define the transforms the images will go through
tf = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(ds_mean,ds_std)
])
# Load the training and test dataset/dataloader
torch.manual_seed(100)
# reapply tf
cifar_train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = tf)
train_set, valid_set = torch.utils.data.random_split(cifar_train_set,[45000,5000])

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 256, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = 256, shuffle = False)

test_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = False, transform = tf)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 256, shuffle = False)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# CNN model

class cifarnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.d = nn.Dropout(.2)
        self.conv2 = nn.Conv2d(64,128,3,1,1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128,256,3,1,1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256,512,3,1,1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(512*2*2,10)
        #self.fc2 = nn.Linear(128,10)
        #self.fc2 = nn.Linear(128,256)
        #self.fc3 = nn.Linear(256,512)
        #self.fc4 = nn.Linear(512,1024)
        #self.fc5 = nn.Linear(1024,10)
        
    def forward(self,x):
        x = self.d(self.pool(self.bn1(F.relu(self.conv1(x)))))
        x = self.d(self.pool(self.bn2(F.relu(self.conv2(x)))))
        #print(x.shape)
        x = self.d(self.pool(self.bn3(F.relu(self.conv3(x)))))
        x = self.d(self.pool(self.bn4(F.relu(self.conv4(x)))))
        x = torch.flatten(x,1)
        x = self.fc1(x)
        
        return x
model = cifarnet()
model.to(device)
# loss function and optimizer

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr = 0.008)

# Calc accuracy

def accuracy(data_loader,dl_type):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            output = model(images)
            _, predicted = torch.max(output.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on {len(data_loader)*256} {dl_type} images: {100 * correct // total} %')
    
# Train the network
epochs = 10
best_val_loss = np.inf
for epoch in range(epochs):
    total_loss = 0
    print(f'Epoch {epoch+1}: \n')
    for i, data in enumerate(train_loader):
        # Retrieve inputs
        images, labels = data[0].to(device), data[1].to(device)
        
        #clear gradient
        optimizer.zero_grad()
        
        #forward step
        output = model(images)
        loss = criterion(output,labels)
        
        #backward step
        loss.backward()
        
        #optimize
        optimizer.step()
        total_loss += loss.item()
       
    valid_loss = 0
    accuracy(train_loader,"train")
    for j, vdata in enumerate(valid_loader):
        vimages, vlabels = vdata
        output = model(vimages)
        loss = criterion(output,vlabels)
        valid_loss += loss.item()
    print(f'Epoch {epoch+1}: \n Training Loss: {total_loss/len(train_loader)} \n Validation Loss: {valid_loss/len(valid_loader)}')
    accuracy(valid_loader,"validation")
    if best_val_loss > valid_loss:
        best_val_loss = valid_loss
        print("Saving best model so far \n")
        torch.save(model.state_dict(), 'cifar_model_10e_with_val.pth')

# Test network
correct = 0
total = 0

# no gradients necessary when testing

accuracy(test_loader,"test")

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

PATH = './final_model2.pth'
#torch.save(model.state_dict(), PATH)