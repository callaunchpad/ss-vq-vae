import time
import copy
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from RNN_dataloader import *
from LSTM_class import *

input_size = 2048
embedding_size = 1024
hidden_size = 512


genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
class_label_to_name = {i: genres[i] for i in range(len(genres))}

# download the data

model = LSTMNetwork(input_size, embedding_size, hidden_size)

# model.load_state_dict(torch.load('trained_model'))
model.eval()
#print(torch.__version__)

def train_model(model, dataloaders, criterion, num_epochs=25):
    """Train a model and save best weights
    
    Adapted From:
        https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    
    Args:
        model (nn.Module): Model to train
        dataloaders ({"train": dataloader, "val": dataloader}): Training dataloader and validation dataloader
        criterion (function): Loss function
        num_epochs (int, optional): Number of epochs to train for. Defaults to 25.
    Returns:
        (model, validation_accuracy): Model with best weights, Array of validation loss over training
    """
    since = time.time()

    # optimizer only updates parameters that are un-frozen
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    # Useful if your environment supports CUDA; don't worry about it if it doesn't
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    val_acc_history = []
    training_acc_history = []
    val_loss_history = []
    training_loss_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            best_acc = 0.0

            state_h, state_c = model.init_hidden_states(batch_size)

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    logits, (state_h, state_c) = model(inputs, (state_h, state_c))
                    loss = criterion(logits.transpose(1,2), labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            else:
                training_acc_history.append(epoch_acc)
                training_loss_history.append(epoch_loss)

    return model, val_acc_history, training_acc_history, val_loss_history, training_loss_history

def visualize_model(model, dataloaders, num_songs=5):
    # Utility function for visualizing predictions
    print("IN VISUALIZE MODEL")
    was_training = model.training
    model.eval()
    songs_so_far = 0
    
    # Useful if your environment supports CUDA; don't worry about it if it doesn't
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs, labels = inputs.to(device), labels.to(device)
            logoutputs = model(inputs)
            _, preds = torch.max(logoutputs, 1)

            # plotting songs for train/val history
            for j in range(inputs.size()[0]):
                songs_so_far += 1
                outputs = torch.exp(logoutputs) # convert to probabilities by taking log
                outputs = outputs.cpu().data.numpy().squeeze()
                
                # print(outputs[j])
                # print(preds[j])
                fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
                true_label = labels.cpu().data.numpy()[j]
                ax1.set_title(f'predicted: {class_label_to_name[preds[j].item()]} | actual: {class_label_to_name[labels[j].item()]}')
                
                ax1.axis('off')
                
                ax2.barh(np.arange(10), outputs[j])
                ax2.set_aspect(0.1)
                ax2.set_yticks(np.arange(10))
                ax2.set_yticklabels(genres)
                ax2.set_title('Class Probability')
                ax2.set_xlim(0, 1.1)
                plt.tight_layout()
                plt.show()
                if songs_so_far == num_songs:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# train the model

# Create Data Loaders
# -- these will efficiently load your cifar-10 data 32 images at a time
batch_size = 25

init_dataset = RNNDataset()

split_lens = [2, 1]
print(split_lens)
# dataset, val_dataset = data.random_split(init_dataset, split_lens, generator=torch.Generator().manual_seed(42))
dataset, val_dataset = data.random_split(init_dataset, split_lens)

    
train_data_loader = torch.utils.data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)

dataloaders = {'train':train_data_loader, "val": val_data_loader}

# Loss function for classification
criterion = nn.NLLLoss()

# Call train model function you defined above

trained_model, val_acc, training_acc, val_loss, training_loss = train_model(model, dataloaders, criterion, num_epochs=20)

torch.save(trained_model.state_dict(), 'trained_model')
# Visualize validation loss and some predictions
# plt.plot(range(len(val_acc)), val_acc, label='validation')
# plt.plot(range(len(training_acc)), training_acc, label='training')
plt.plot(range(len(val_loss)), val_loss, label='validation loss')
plt.plot(range(len(training_loss)), training_loss, label='training loss')
plt.legend()


plt.title("Classifier Loss vs. Epoch")
plt.show()
plt.savefig('accuracy3.png')


# Plots predictions from trained cnn
visualize_model(trained_model, dataloaders, num_songs=5)

def prediction_accuracy(model, dataloaders):
    """Computes accuracy on train and validation set"""
    since = time.time()
    
    model.eval()   # Set model to evaluate mode
    
    loss = {}
    acc = {}
    
    with torch.no_grad():
        for phase in ['train', 'val']:
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # move to GPU if possible
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                
                # statistics
                running_corrects += torch.sum(preds == labels.data)

            acc[phase] = running_corrects.double() / len(dataloaders[phase].dataset)


    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Training Accuracy: ", acc['train'].item(), "Validation Accuracy: ", acc['val'].item())
    
    return acc

prediction_accuracy(trained_model, dataloaders)
