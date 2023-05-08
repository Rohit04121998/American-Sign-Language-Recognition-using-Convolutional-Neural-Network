American Sign Language Recognition using Convolutional Neural Network using ASL Dataset

This repository houses the official PyTorch implementation of USC's EE 641 (Deep Learning System) final project. All the modules have been well-documented and are self-explanatory. However, for any clarifications regarding the code published here, please feel free to reach out to us at anantapa@usc.edu or veeradhi@usc.edu with the subject line as "EE 541 Project", for consideration.

Abstract : - 
ASL is a language used by the deaf and hard-of-hearing communities based on visual gestures. Recognizing ASL signs is difficult due to their complex and variable nature. However, CNNs have been trained on large datasets of annotated ASL images to accurately recognize signs. These
networks use a hierarchical approach to learn local features in different parts of the input image and combine them to identify the overall sign. Another application of CNNs in ASL is translating sign language into written or spoken language. This task involves mapping signs in an image to their corresponding English words or sentences. CNNs have been combined with RNNs to encode the video frames and decode the output text. ASL can improve communication with the deaf and hard-of-hearing communities across various industries. For example, using ASL can lead to better patient satisfaction and outcomes in healthcare. Academic research on ASL can also enhance our understanding of the language, its structure, and its applications, leading to more efficient teaching approaches and learning resources for those who are learning ASL.

For Installation Setup - 
$ git clone https://github.com/Rohit04121998/EE541_project.git
$ cd EE541_project

The utils folder consists of python files required for the models. 
1. data.py consists of the function to split the dataset.
2. evaluate.py consists of two functions. The first function evaluates the test data and the second function prints the confusion matrix.
3. models.py consists of 2 CNN architectures used in this project.
4. plot.py consists of function to plot the accuracy and loss scurves.
5. train.py consists of functions to train the model. 2 extra functions are defined for the single step forward and backward pass of network.

Contact Details -

Venkat Giri Anantapatnaikuni  - anantapa@usc.edu

Rohit Veeradhi - veeradhi@usc.edu
