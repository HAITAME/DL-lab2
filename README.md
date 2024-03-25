# Deep Learning Lab: Computer Vision Models

This repository contains the implementation of various computer vision models using PyTorch for the MNIST dataset. The objective of this lab is to explore different neural architectures such as CNN, Faster R-CNN, VGG16, AlexNet, and Vision Transformers (ViT).

## Part 1: CNN Classifier and Faster R-CNN

In this part, we establish and compare the performance of two different models: a Convolutional Neural Network (CNN) classifier and Faster R-CNN for object detection on the MNIST dataset.

### CNN Classifier
- **Architecture**: Define a CNN architecture with convolutional layers, pooling layers, and fully connected layers.
- **Hyperparameters**: Define kernel sizes, padding, stride, optimizers, regularization, etc.
- **Implementation**: Train the CNN model on the MNIST dataset using PyTorch, running on GPU.
- **Evaluation Metrics**: Compare performance using accuracy, F1 score, loss, and training time.

### Faster R-CNN
- **Architecture**: Implement Faster R-CNN for object detection on the MNIST dataset.
- **Hyperparameters**: Configure the model and training parameters.
- **Implementation**: Train the Faster R-CNN model and evaluate its performance.
- **Comparison**: Compare results with the CNN classifier based on various metrics.

### Fine-Tuning with VGG16 and AlexNet
- **Fine-Tuning**: Retrain VGG16 and AlexNet models on the MNIST dataset.
- **Comparison**: Compare the performance of fine-tuned models with CNN and Faster R-CNN.

## Part 2: Vision Transformer (ViT)

In this part, we explore the Vision Transformer (ViT) architecture and its performance on the MNIST dataset.

### Vision Transformer (ViT)
- **Tutorial**: We followed a tutorial ([Vision Transformers from Scratch: PyTorch - A Step-by-Step Guide](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)) to establish a ViT model architecture from scratch.
- **Implementation**: Perform image classification tasks on the MNIST dataset.

## Links
- [MNIST Dataset Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
