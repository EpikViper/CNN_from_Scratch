import numpy as np 

class Model:
    def __init__(self, loss, optimizer):
        self.input = None 
        self.output = None 
        self.optimizer = optimizer
        self.loss = loss
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        self.input = X
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)

        self.output = current_input

        return self.output

    def fit(self, iters, lr, X, Y):
        loss_history = []
        self.optimizer.initialize_layers(self.layers)
        
        for i in range(iters):
            Y_hat = self.forward(X)
            loss_i = self.loss.calculate(Y_hat, Y)
            loss_history.append(loss_i)

            output_error = self.loss.derivative(Y_hat, Y)

            for layer in reversed(self.layers):
                output_error = layer.backward(output_error)
                if hasattr(layer, 'W'):
                    dW, dB = self.optimizer.compute_gradient(layer, layer.dW, layer.dB)
                    layer.W -= lr * dW
                    layer.B -= lr * dB

        print(loss_history)


    def predict(self, X):

        result = self.forward(X)
        print(result)


