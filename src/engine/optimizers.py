import numpy as np 

class SGD:
    def __init__(self):
        pass 

    def initialize_layers(self, layers):
        pass 

    def compute_gradient(self, layer, dW, dB):
        
        return dW, dB 


class Momentum:
    def __init__(self, p):
        self.p = p
        self.momentum = {}

    def initialize_layers(self, layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                self.momentum[layer] = {
                    'mW': np.zeros_like(layer.W),
                    'mB': np.zeros_like(layer.B)
                }

    def compute_gradient(self, layer, dW, dB):
        p = self.p
        k = self.momentum[layer]
        k['mW'] = p * k['mW'] + (1-p) * dW 
        k['mB'] = p * k['mB'] + (1-p) * dB


        return k['mW'], k['mB']


class RMSprop:
    def __init__(self, p):
        self.velocity = {}
        self.beta = p 


    def initialize_layers(self, layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                self.velocity[layer] = {
                    'vW': np.zeros_like(layer.W),
                    'vB': np.zeros_like(layer.B)
                }


    def compute_gradient(self, layer, dW, dB):
        p = self.beta
        k = self.velocity[layer]

        k['vW'] = p * k['vW'] + (1-p) * (dW**2)
        k['vB'] = p * k['vB'] + (1-p) * (dB**2)

        dW = dW / np.sqrt(k['vW'] + 1e-10)
        dB = dB / np.sqrt(k['vB'] + 1e-10)

        return dW, dB



class Adam:
    def __init__(self, p1, p2):
        self.velocity = {}
        self.momentum = {}
        self.beta_m = p1 
        self.beta_v = p2
        self.t = 0
    

    def initialize_layers(self, layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                self.velocity[layer] = {
                    'vW': np.zeros_like(layer.W),
                    'vB': np.zeros_like(layer.B)
                }
                self.momentum[layer] = {
                    'mW': np.zeros_like(layer.W),
                    'mB': np.zeros_like(layer.B)
                }

    
    def compute_gradient(self, layer, dW, dB):
        self.t += 1
        t = self.t
        m = self.momentum[layer]
        v = self.velocity[layer]
        p1 = self.beta_m
        p2 = self.beta_v

        v['vW'] = p2 * v['vW'] + (1-p2) * (dW**2)
        v['vB'] = p2 * v['vB'] + (1-p2) * (dB**2)
        m['mW'] = p1 * m['mW'] + (1-p1) * dW 
        m['mB'] = p1 * m['mB'] + (1-p1) * dB

        vW_temp = v['vW'] / (1 - p2**t)
        mW_temp = m['mW'] / (1 - p1**t)
        vB_temp = v['vB'] / (1 - p2**t)
        mB_temp = m['mB'] / (1 - p1**t)

        dW = mW_temp / (np.sqrt(vW_temp) + 1e-10)
        dB = mB_temp / (np.sqrt(vB_temp) + 1e-10)

        return dW, dB 
