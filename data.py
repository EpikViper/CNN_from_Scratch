import numpy as np 
import zipfile 

with zipfile.ZipFile('MNIST.zip', 'r') as zip_ref:
    with zip_ref.open('train-images.idx3-ubyte') as x, \
         zip_ref.open('train-labels.idx1-ubyte') as y:
        X = x.read()
        Y = y.read()

def load_mnist_images(file):
    data = np.frombuffer(file, dtype=np.uint8, offset=16)
    
    return data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

def load_mnist_labels(file):
    
    data = np.frombuffer(file, dtype=np.uint8, offset=8)
    formatted = np.zeros((data.size, 10))
    formatted[np.arange(data.size), data] = 1
    return formatted


X_train = load_mnist_images(X)
Y_train = load_mnist_labels(Y)

# print(Y_train[:10])