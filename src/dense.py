import numpy as np 


class Dense:
    def __init__(self, n_in, n_out):
        limit = np.sqrt(6 / (n_in + n_out))
        self.input = None 
        self.W = np.random.uniform(-limit, limit, size=(n_in, n_out))
        self.B = np.zeros((1, n_out))

    def forward(self, input):
        self.input = input

        return np.matmul(input, self.W) + self.B 

    
    def backward(self, output_error):
        m = len(self.input)

        dW = np.matmul(self.input.T, output_error) / m 
        dB = np.sum(output_error, axis=0, keepdims=True) / m 

        input_error = np.matmul(output_error, self.W.T)

        return input_error
