import numpy as np 

class SoftmaxCCE:
    def __init__(self, n_in):
        pass

    def forward(self, input):
        max = np.max(input, axis=0, keepdims=True)
        exp = np.exp(input - max)

        return exp / np.sum(exp, axis=0, keepdims=True)

    
    def backward(self, output_error):
        # since this is always paired with CCE, I will combine the derivative. 
        # I'll write dummy here and implement Y_hat - Y in CCE later on. 
        dummy_derivative = 1 

        input_error = output_error * dummy_derivative

        return input_error

