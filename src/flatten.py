import numpy as np

class Flatten():
    def __init__(self):
        self.output = None 
        self.shape = None


    def forward(self, input):
        self.shape = input.shape
        size = input.size
        self.output = input.reshape(size, 1)

        return self.output

    def backward(self, output_error):

        input_error = output_error.reshape(self.shape)

        return input_error
