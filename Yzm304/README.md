# Banknote Authentication using Multi-Layer Perceptrons

## Introduction
This project implements and compares different Multi-Layer Perceptron (MLP) architectures for banknote authentication using both NumPy and PyTorch implementations. 
The goal is to accurately classify banknotes as genuine or counterfeit based on various extracted features.

### Problem Statement
Banknote authentication is a critical security task that requires high accuracy and balanced performance between false positives and false negatives. 
The challenge is to develop a reliable neural network model that can effectively distinguish between genuine and counterfeit banknotes.

## Methods

### Dataset
- **Source**: UCI Machine Learning Repository's Banknote Authentication dataset
- **Features**: 4 input features (Variance, Skewness, Curtosis, Entropy)
- **Target**: Binary classification (0: Genuine, 1: Counterfeit)
- **Split**: 60% Training, 20% Validation, 20% Test

### Model Architectures
1. **Two-Layer MLP**:
   - Input Layer: 4 neurons
   - Hidden Layer: 32 neurons
   - Output Layer: 1 neuron (sigmoid activation)

2. **Three-Layer MLP**:
   - Input Layer: 4 neurons
   - First Hidden Layer: 32 neurons
   - Second Hidden Layer: 32 neurons
   - Output Layer: 1 neuron (sigmoid activation)

### Implementation Details
- **Frameworks**: NumPy (from scratch) and PyTorch
- **Activation Functions**: ReLU and Tanh variants
- **Learning Rate**: 0.0001
- **Decision Threshold**: 0.6
- **Early Stopping**: Patience of 50 epochs
- **Loss Function**: Binary Cross-Entropy

## Results

### Performance Metrics
- Accuracy, Precision, Recall, and F1-score for each model
- Confusion matrices showing true/false positives/negatives
- Training and validation loss curves

### Key Findings
1. Model Architecture Comparison:
   - Two-layer vs. Three-layer performance
   - ReLU vs. Tanh activation effects
   - NumPy vs. PyTorch implementation differences

2. Optimization Choices:
   - Effect of threshold adjustment on false positives/negatives
   - Impact of learning rate on convergence
   - Early stopping behavior

## Discussion
It seems that under the same conditions the numpy method need more epochs to get a slighty worse result.
Turns out a library made for deep learning methods such as Pyhtorch performs better then an unoptimized numpy implemantation.
Upon further optimization and usage of other methods the results may come closer to the pytorch one however this assingment did not require that therefore i saw no point in doing so.
