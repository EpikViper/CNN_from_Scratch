import numpy as np 

class Conv2D:
    def __init__(self, n_filters, kernel_size, input_channels):
        m, n = kernel_size
        n_in = m * n * input_channels
        n_out = m * n * n_filters
        limit = np.sqrt(6 / (n_in + n_out))
        self.W = np.random.uniform(-limit, limit, size=(n_filters, input_channels, m, n))
        self.B = np.zeros((1, n_filters))
        self.filters = n_filters
        self.input = None
        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)


    def forward(self, input):
        _, _, m, n = self.W.shape

        if input.ndim != 4:
            input = input[np.newaxis, :, :, :]

        self.input = input
        b, c, h, w = input.shape

        H_out =  h - m + 1
        W_out = w - n + 1

        output = np.zeros((b, self.filters, H_out, W_out)) + self.B.reshape(1, self.filters, 1, 1)

        for i in range(m):
            for j in range(n):
                input_slice = self.input[:, :, i:i+H_out, j:j+W_out]

                output += np.einsum('bchw,oc->bohw', input_slice, self.W[:,:,i,j])

        self.output = output

        return self.output



    def backward(self, output_error):
        n_batch, n_filters, H_out, W_out = output_error.shape 
        _, _, m, n = self.W.shape

        dW = np.zeros_like(self.W)
        dB = np.zeros_like(self.B) 
        dX = np.zeros_like(self.input)

        for i in range(m):
            for j in range(n):
                weight_slice = self.W[:,:,i,j]
                input_slice = self.input[:, :, i:i+H_out, j:j+W_out]

                dW[:,:,i,j] += np.einsum('bohw,bchw->oc', output_error, input_slice)
                dX[:, :, i:i+H_out, j:j+W_out] = np.einsum('bohw,oc->bchw', output_error, weight_slice)

        self.dB = np.sum(output_error, axis=(0,2,3), keepdims=True)

        dW = dW / n_batch
        dB = dB / n_batch

        self.dW = dW 
        self.dB = dB 

        return dX


