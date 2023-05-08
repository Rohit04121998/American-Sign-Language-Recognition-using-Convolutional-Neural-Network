import torch
import numpy as np
import torch.nn as nn
import time

def train_model(model, train_loader, val_loader, learning_rate, epochs, device, reg=0):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    
    total_duration = 0
    epoch_list = []

    for epoch in range(epochs):
        start_time = time.time()
        
        train(train_loader, model, loss_func, optimizer, device)
        train_loss, train_accuracy = val(train_loader, model, loss_func, device)
        val_loss, val_accuracy = val(val_loader, model, loss_func, device)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss_list.append(val_loss)
        test_accuracy_list.append(val_accuracy)

        cur_duration = time.time() - start_time
        total_duration += cur_duration
        epoch_list.append(cur_duration)

        print('Epoch {} -> Loss = {:.4f} | Train Accuracy = {:.2f}% | valation Accuracy = {:.2f}%'.format(epoch+1, train_loss, train_accuracy, val_accuracy))

    print('-'*100)
    print('Time taken to train: {:.2f}s'.format(total_duration))
    print('Average time of each epoch: {:.2f}s'.format(np.mean(epoch_list)))

    return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list

def train(data_loader, model, loss_func, optimizer, device):
    model.train()

    for inputs, labels in data_loader: 
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def val(data_loader, model, loss_func, device):
    size = len(data_loader.dataset)
    batches = len(data_loader)
    total_loss = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            outputs, labels = outputs.cpu(), labels.cpu()

            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred==labels).sum().numpy()
            
    total_loss /= batches
    accuracy = (correct/size)*100

    return total_loss, accuracy