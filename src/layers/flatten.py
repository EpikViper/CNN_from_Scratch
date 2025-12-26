import numpy as np

class Flatten():
    def __init__(self):
        self.shape = None


    def forward(self, input):
        self.shape = input.shape
        n_batch = input.shape[0]

        # reshapes to (n_batch, everything else)
        return input.reshape(n_batch, -1)

    def backward(self, output_error):

        input_error = output_error.reshape(self.shape)

        return input_error
