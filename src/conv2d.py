import numpy as np 

class Conv2D:
    def __init__(self, n_filters, kernel_size, input_channels):
        m, n = kernel_size
        n_in = m * n * input_channels
        n_out = m * n * n_filters
        limit = np.sqrt(6 / (n_in + n_out))
        self.W = np.random.uniform(-limit, limit, size=(n_filters, input_channels, m, n))
        self.B = np.zeros((1, n_filters))
        self.channels = input_channels
        self.filters = n_filters
        self.input = None
        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)


    def forward(self, input):
        self.input = input
        m = self.W.shape[2]
        n = self.W.shape[3]
        n_batch, c_n, h, w = input.shape
        W_out =  h - m + 1
        H_out = w - n + 1

        output = np.zeros((n_batch, self.filters, W_out, H_out))

        for b in range(n_batch):
            for f in range(self.filters):
                for i in range(W_out):
                    for j in range(H_out):
                        current = 0
                        for c in range(c_n):
                            current += np.sum(self.W[f][c] * self.input[b, c, i:i+m, j:j+n])
                    
                    output[b][f][i][j] = current + self.B[0, f]

        return output


    def backward(self, output_error):
        n_batch, n_filters, h, w = output_error.shape 
        n_filters, n_channels, m, n = self.W.shape

        dW = np.zeros_like(self.W)
        dB = np.zeros_like(self.B) 
        dX = np.zeros_like(self.input)

        for b in range(n_batch):
            for f in range(n_filters):
                # sums errors from all filter maps, for each example
                dB[0, f] += np.sum(output_error[b, f])

                for i in range(h):
                    for j in range(w):

                        error = output_error[b, f, i, j]

                        dW[f] += error * self.input[b, :, i:i+m, j:j+n] 

                        dX[b, :, i:i+m, j:j+n] += error * self.W[f]

        dW = dW / n_batch
        dB = dB / n_batch

        self.dW = dW 
        self.dB = dB 

        return dX


