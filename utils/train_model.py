import time
import copy
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def train_model(model, dataloaders, criterion, optimizer, model_file, num_epochs=25, is_inception=False, mode='train'):
    """
    Modified version of https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    assert mode=='train' or mode=='test', "Mode must be one of: 'train', 'test'"
    
    since = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if mode == 'train':
        phases = ['train', 'val']
        train_acc_history = []
        train_loss_history = []

    if mode == 'test':
        phases = ['val']
        num_epochs = 1       
        
    val_acc_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
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
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print(mode + 'ing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    if mode == 'train':
        print('Best val Acc: {:4f}'.format(best_acc))
    
    if mode == 'train':
        # load and save best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), model_file)
        stats = (train_acc_history, train_loss_history, val_acc_history, val_loss_history)
    
    if mode == 'test':
        stats = (val_acc_history, val_loss_history)
        
    return model, stats
    
   
    
def plot_stats(train_acc, train_loss, val_acc, val_loss):
    figure(figsize=(8, 6))
    plt.plot(train_acc, label="Training accuracy")
    plt.plot(val_acc, label="Validation accuracy")
    plt.legend()
    plt.show()
    figure(figsize=(8, 6))
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.legend()
    plt.show()
    
def plot_stats_compare(train_acc, train_loss, val_acc, val_loss, train_acc_sc, train_loss_sc, val_acc_sc, val_loss_sc):
    figure(figsize=(8, 6))
    plt.plot(train_acc, label="Training accuracy, pretrained")
    plt.plot(val_acc, label="Validation accuracy, pretrained")
    plt.plot(train_acc_sc, label="Training accuracy, from scratch")
    plt.plot(val_acc_sc, label="Validation accuracy, from scratch")
    plt.legend()
    plt.show()
    figure(figsize=(8, 6))
    plt.plot(train_loss, label="Training loss, pretrained")
    plt.plot(val_loss, label="Validation loss, pretrained")
    plt.plot(train_loss_sc, label="Training loss, from scratch")
    plt.plot(val_loss_sc, label="Validation loss, from scratch")
    plt.legend()
    plt.show()