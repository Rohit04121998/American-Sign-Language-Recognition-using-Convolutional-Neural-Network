import numpy as np
import matplotlib.pyplot as plt

def db(x):
    return 20*np.log10(x)

def plot_metrics(train_loss, train_acc, val_loss, val_acc):
    print('Final Train Accuracy = {:.2f}%'.format(train_acc[-1]))
    print('Final Validation Accuracy  = {:.2f}%'.format(val_acc[-1]))

    epochs = np.arange(1, len(train_loss)+1)
    plt.plot(epochs, db(train_loss), label='Train set')
    plt.plot(epochs, db(val_loss), label='Validation set')
    plt.title("Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss (dB)")
    plt.show()

    plt.plot(epochs, train_acc, label='Train set')
    plt.plot(epochs, val_acc, label='Validation set')
    plt.title("Accuracy Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.show()