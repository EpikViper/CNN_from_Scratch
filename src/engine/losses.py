import numpy as np 

class MSE:
    def __init__(self):
        pass 


    def calculate(self, Y_hat, Y):

        return np.mean(np.sum((Y_hat - Y)**2)).item() / 2

    def derivative(self, Y_hat, Y):
        
        return Y_hat - Y


class CCE:
    def __init__(self):
        pass

    def calculate(self, Y_hat, Y):

        loss = np.sum(-Y * np.log(Y_hat + 1e-10), axis=1)

        return np.mean(loss).item()


    # using this in combination with SoftmaxCCE layer, whose derivative is 1 (works in combo)
    def derivative(self, Y_hat, Y):

        return Y_hat - Y


