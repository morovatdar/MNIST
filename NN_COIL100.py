import torch
import torchvision
import numpy as np
from torch import nn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
from torchvision.io import read_image

class COIL100(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.labels_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        image = read_image(self.img_dir + '/' + self.labels_list[idx]).numpy()
        
        f = self.labels_list[idx]
        if f[4:6] == "__": label = int(f[3])-1
        elif f[5:7] == "__": label = int(f[3:5])-1
        else: label = int(f[3:6])-1

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def load_data(batch_size, test_batch_size):
    """ Load dataset from the torchvision library (a pytorch library)
        Arguments: 
            batch_size: An integer defining the number of training instances in a batch.
            test_batch_size: An integer defining the number of testing instances in a batch.
        Returns: 
            train_dataloader: An iterable over MNIST training dataset.
            test_dataloader: An iterable over MNIST testing dataset.
    """
    img_dir = "C:\SDSMT\Thesis\Code\Dataset\coil-100"
    # PyTorch dataset
    dataset = COIL100(img_dir, transform=T.ToTensor())
    
    test_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - test_size
    
    training_data, test_data = random_split(dataset, [train_size, test_size])

    # Wraps iterables around the Datasets to enable easy access to the instances.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader


class NeuralNetwork(nn.Module):
    """ A subclass of nn.Module to define different layers of the neural network
        Methods: 
             __init__: initialize the neural network layers.
            forward: implements the operations on the input dataset.
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*128*128, 512), # First hidden layer with 512 nodes
            nn.ReLU(),
            nn.Linear(512, 512), # Second hidden layer with 512 nodes
            nn.ReLU(),
            nn.Linear(512, 100),) # Output layer with 10 nodes each representing one digit

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    """ Train the model with one given epoch
        Arguments: 
            dataloader: The iterable over the MNIST training dataset.
            model: The defined neural network model.
            loss_fn: Loss function.
            optimizer: It update the model parameter.
        Returns: 
            train_correct: float scalar- Ratio of correct predicted digits to total digits.
            train_loss: float scalar- Average training loss over all batches in one epoch.
    """
    model.train()
    train_loss, train_correct = 0, 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device) # Move variables to GPU/CPU
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        train_loss += loss.item()
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader) # dividing by number of batches
    train_correct /= len(dataloader.dataset) # dividing by the size of dataset
    return train_correct, train_loss


def test_loop(dataloader, model, loss_fn):
    """ Test the model with one given epoch
        Arguments: 
            dataloader: The iterable over the MNIST test dataset.
            model: The defined neural network model.
            loss_fn: Loss function.
        Returns: 
            test_correct: float scalar- Ratio of correct predicted digits to total digits.
            test_loss: float scalar- Average testing loss over all batches in one epoch.
    """
    model.eval()
    test_loss, test_correct = 0, 0

    # Doing the forward step by disabling the gradient calculation 
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            test_loss += loss_fn(pred, y).item()
            test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= len(dataloader)
    test_correct /= len(dataloader.dataset)
    print(f"Accuracy: {(100*test_correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_correct, test_loss


def test_instances(dataloader, model, loss_fn):
    """ Test the model for the given test dataset
        Arguments: 
            dataloader: The iterable over the MNIST test dataset.
            model: The defined neural network model.
            loss_fn: Loss function.
        Returns: 
            accuracy: float vector- accuracy of classification for each digit
            batch_correct: float vector- Ratio of correct predicted digits to total digits for each batch.
            batch_loss: float vector- The average testing loss for each batch.
    """
    # Initializing the number of correct digits and the total number of digits
    model.eval()
    correct, total = torch.zeros(100).to(device), torch.zeros(100).to(device)
    batch_loss = torch.zeros(len(dataloader))
    batch_correct = torch.zeros(len(dataloader))
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            batch_loss[batch] = loss_fn(pred, y).item()
            batch_correct[batch] = (pred.argmax(1) == y).type(torch.float).sum().item()/len(y)
            
            # Calculating the number of correct digits for each digit in each batch
            for i in range(100):
                pred_i, y_i = torch.clone(pred.argmax(1)), torch.clone(y) # Make a copy of data
                # Assigning an out of scope digit to unmatched digits
                pred_i[pred_i != i] = -1 
                y_i[y_i != i] = -2
                correct[i] += (pred_i == y_i).type(torch.float).sum().item()
            
            count = torch.bincount(y) # Count all the digits in each batch
            # Pad the count vector for the last batch when 9 or 8 may not present beacuse of the small batch size
            if len(count) < 100:
                count = torch.cat((count, torch.zeros(100-len(count)).to(device)), dim=0)
            total += count
            
    accuracy = torch.div(correct, total)
    return torch.Tensor.cpu(accuracy), batch_loss, batch_correct

   
if __name__ == '__main__':
    # Setting hyperparameter:
    # Optimizer parameters
    learning_rate = 1e-2
    momentum = 0.90
    gamma = 0.7
    # Loop parameters
    batch_size = 64
    test_batch_size = 1000
    epochs = 30

    random_seed = 11
    torch.manual_seed(random_seed)

    # Selecting GPU as the calculation device, if available
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"

    # Initialize the model and datasets
    model = NeuralNetwork().to(device) 
    train_dataloader, test_dataloader = load_data(batch_size, test_batch_size)

    # Initialize the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    
    # Initialize the model evaluation parameters
    test_correct, test_loss = torch.zeros(epochs), torch.zeros(epochs)
    train_correct, train_loss = torch.zeros(epochs), torch.zeros(epochs)

    # Repeat the training loop for epochs times
    for t in range(epochs):
        print(f"Epoch {t+1}--->", end =" ")
        train_correct[t], train_loss[t] = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_correct[t], test_loss[t] = test_loop(test_dataloader, model, loss_fn)
        scheduler.step()

    print(f"Epoch {t+1}--->")
    print(f"Training Accuracy: {(100*train_correct[t]):>0.1f}%,   Training Avg. loss: {train_loss[t]:>8f}")
    print(f"Testing Accuracy: {(100*test_correct[t]):>0.1f}%,   Testing Avg. loss: {test_loss[t]:>8f}")
    
    # Plotting the loss function vs epochs
    plt.plot(range(1, epochs+1), train_loss)
    plt.plot(range(1, epochs+1), test_loss)
    plt.title('MLP NeuralNetwork Learning Curve (loss)')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # Plotting the model accuracy vs epochs
    plt.plot(range(1, epochs+1), train_correct)
    plt.plot(range(1, epochs+1), test_correct)
    plt.title('MLP NeuralNetwork Learning Curve (accuracy)')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Classification accuracy')
    plt.show()

    # Testing the final model
    correct_digits, batch_loss, batch_correct = test_instances(test_dataloader, model, loss_fn)
    
    # Plotting the model accuracy vs different digits
    fig, ax = plt.subplots()
    bars = ax.bar(range(100), torch.round(1000 * correct_digits)/1000)
    plt.title('MLP NeuralNetwork Accuracy in Classification of Objects')
    plt.xlabel('Objects')
    plt.ylabel('Classification accuracy')
    plt.show()
    
    # Plotting box and whisker for loss and accuracy results of batches
    
    plt.subplot(1, 2, 1)
    plt.boxplot([batch_correct])
    plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.boxplot([batch_loss])
    plt.title('Loss')
    plt.suptitle('MLP NeuralNetwork Box and Whisker Plots')
    plt.show()

    # Save the model for future use
    torch.save(model.state_dict(), './NN_COIL100.pth')
