import numpy as np 

class ReLU:
    def __init__(self):
        self.output = None


    def forward(self, input): 
        self.output = np.where(input > 0, input, 0)

        return self.output 

    def backward(self, output_error):
        derivative = np.where(self.output > 0, 1, 0)

        input_error = derivative * output_error

        return input_error