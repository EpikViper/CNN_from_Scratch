import numpy as np 

class MaxPooling2D:
    def __init__(self, filter_size, input_channels):
        self.input = None
        self.filter = filter_size

    
    def forward(self, input):
        self.input = input
        n_batch, n_channels, h, w = self.input.shape
        x, y = self.filter
        m = int(h / x)
        n = int(w / y)
        output = np.zeros((n_batch, n_channels, m, n))

        for i in range(m):
            for j in range(n):
                patch = self.input[:,:, x*i:x*(i+1), y*j:y*(j+1)]
                max_in_patch = np.max(patch, axis=(2,3))
                output[:,:,i,j] = max_in_patch


        return output


    def backward(self, output_error):
        n_batch, n_channels, m, n = output_error.shape 
        n_batch, n_channels, a, b = self.input.shape
        x,y = self.filter

        input_error = np.zeros_like(self.input)

        for i in range(m):
            for j in range(n):
                pixel = output_error[:,:,i,j]
                patch = self.input[:,:, x*i:x*(i+1), y*j:y*(j+1)]
                max_in_patch = np.max(patch, axis=(2,3), keepdims=True)

                # compares every element in patch. Puts 1 where element = maxium and 0s elsewhere.
                mask = (patch == max_in_patch)

                # since we need to multiply this error by mask, it needs 4 dimensions, so we reshape it to match.
                error_i_j = output_error[:, :, i, j].reshape(n_batch, n_channels, 1, 1)

                input_error[:, :, x*i:x*(i+1), y*j:y*(j+1)] = mask * error_i_j
                          

        return input_error
            