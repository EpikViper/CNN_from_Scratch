import numpy as np 
from src.engine.model import Model
from src.engine.optimizers import Adam
from src.engine.losses import CCE
from src.layers.conv2d import Conv2D
from src.layers.maxpooling2d import MaxPooling2D
from src.layers.dense import Dense
from src.layers.flatten import Flatten 
from src.layers.relu import ReLU
from src.layers.softmax import SoftmaxCCE
from data import X_train, Y_train
import time 


loss = CCE()
optimizer = Adam(0.9, 0.999)
my_model = Model(loss, optimizer)

Conv2D_1 = Conv2D(32, (3,3), 1)
relu_1 = ReLU()
MaxPooling_1 = MaxPooling2D((2,2), 1)
Conv2D_2 = Conv2D(64, (3,3), 32)
relu_2 = ReLU()
MaxPooling_2 = MaxPooling2D((2,2), 1)
flatten_1 = Flatten()
dense_1 = Dense(1600, 128)
relu_3 = ReLU()
dense_2 = Dense(128, 10)
output_layer = SoftmaxCCE(10)

my_model.add(Conv2D_1)
my_model.add(relu_1)
my_model.add(MaxPooling_1)
my_model.add(Conv2D_2)
my_model.add(relu_2)
my_model.add(MaxPooling_2)
my_model.add(flatten_1)
my_model.add(dense_1)
my_model.add(relu_3)
my_model.add(dense_2)
my_model.add(output_layer)


start_time = time.perf_counter()
my_model.fit(50, 1e-3, X_train[:100], Y_train[:100])
end_time = time.perf_counter()
print(end_time - start_time)


