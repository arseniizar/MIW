# Project 3 — Regression

Neural network regression model implemented from scratch using NumPy.

## Usage

```bash
python proj3.py
```

## Features

| Feature | Notes |
|-------|-------|
| Custom Neural Network | Implemented from scratch using NumPy |
| Activation functions | Tanh, Sigmoid, ReLU (using Softplus derivative) |
| Training modes | Batch and Online (Stochastic) |
| Evaluation | Train vs Test loss analysis (overfitting/underfitting) |

## How it works

Data is loaded from the `Dane` directory and split into training and test sets. A custom multi-layer perceptron (MLP) is trained using backpropagation. The model evaluates different activation functions and training methods (batch vs. online). Loss curves are plotted to assess model performance and fitting.