# CNN_from_Scratch

A modular Convolutional Neural Network (CNN) implementation built entirely in NumPy. This project demonstrates the core mechanics of computer vision, including convolutional layers, pooling operations, and spatial data flattening for classification.

# Features

1. **Modular Architecture:** Build deep learning pipelines by stacking `Conv2D`, `MaxPooling2D`, and `Dense` layers using a consistent `.add()` interface.
2. **Convolutional Layers:** Supports 2D convolutions with customizable filters and input channels.
3. **Subsampling:** Includes `MaxPooling2D` to reduce spatial dimensions and control overfitting.
4. **Data Handling:** Integrated `Flatten` layer to transition from 3D feature maps to 1D vectors for dense classification.
5. **Optimization:** Advanced weight updates using the Adam, RMSprop, and Momentum optimizers.

# Project structure

- `model.py`: Core engine managing the forward and backward passes across all layer types.
- `optimizers.py`: Implementation of optimization algorithms like Adam, RMSprop, Momentum and SGD.
- `losses.py`: Suite of loss functions including Categorical Cross-Entropy (CCE), Binary Cross-Entropy (BCE) and Mean Squared Error (MSE).
- `conv2d.py`: 2D Convolutional layer implementation.
- `maxpooling2d.py`: 2D Max Pooling layer implementation.
- `flatten.py`: Utility to flatten multi-dimensional tensors into vectors.
- `dense.py`: Fully connected layer implementation.
- `relu.py`: ReLU activation function.
- `softmax.py`: Combined Softmax and Categorical Cross-Entropy for the output layer.
- `train.py`: Main entry point for model definition, dataset loading, and training execution.

# Usage

```python
import numpy as np
from src.engine.model import Model
from src.engine.optimizers import Adam
from src.engine.losses import CCE
from src.layers.conv2d import Conv2D
from src.layers.maxpooling2d import MaxPooling2D
from src.layers.dense import Dense
from src.layers.flatten import Flatten
from src.layers.relu import ReLU
from src.layers.softmax import SoftmaxCCE

# 1. Initialize Model and Optimizer
loss = CCE()
optimizer = Adam(0.9, 0.999)
my_model = Model(loss, optimizer)

# 2. Add Convolutional and Pooling Layers
my_model.add(Conv2D(32, (3,3), 1))
my_model.add(ReLU())
my_model.add(MaxPooling2D((2,2), 1))

# 3. Add Classification Head
my_model.add(Flatten())
my_model.add(Dense(1600, 128))
my_model.add(ReLU())
my_model.add(Dense(128, 10))
my_model.add(SoftmaxCCE(10))

# 4. Train
# X_train and Y_train should be NumPy arrays
my_model.fit(iters=50, alpha=1e-3, X=X_train, Y=Y_train)
```
