import torch
import random
import operator
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def split_dataset(dataset, num_samples, dataset_split, batch_size, seed=0):
    train_size = int(round(num_samples * dataset_split[0]))
    val_size = int(round(num_samples * dataset_split[1]))
    test_size = int(round(num_samples * dataset_split[2]))
    dataset_index = list(range(len(dataset)))
    
    train_index, val_test_index = train_test_split(dataset_index, train_size=train_size, test_size=test_size+val_size, random_state=seed, stratify=dataset.targets)
    if dataset_split[2] > 0:
        val_index, test_index = train_test_split(val_test_index, train_size=val_size, test_size=test_size, random_state=seed, stratify=operator.itemgetter(*val_test_index)(dataset.targets)) 
    else:
        val_index = val_test_index
    
                                            
    train_set = Subset(dataset, train_index)
    val_set = Subset(dataset, val_index)
    if dataset_split[2] > 0:
        test_set = Subset(dataset, test_index)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    if dataset_split[2] > 0:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

def display_images(dataset, seed=0):
    rows, cols = 2, 3
    idx_to_class = {i: j for j, i in dataset.class_to_idx.items()}
    random.seed(seed)
    index = random.sample(range(len(dataset)), rows*cols)

    fig, ax = plt.subplots(rows, cols)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)

    for i in range(0, rows):
        for j in range(0, cols):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            ax[i, j].imshow(dataset[index[cols*i+j]][0].permute(1, 2, 0))
            ax[i, j].set_title(idx_to_class[dataset[index[cols*i+j]][1]])