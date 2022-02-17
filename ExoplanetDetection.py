# This is a neural net trained to identify exoplanet based on flux data over time 
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import preprocessing

# data set
data = pd.read_csv("exoTrain copy.csv")
data_x = data.drop(["LABEL"], axis = 1)
data_y = data.pop("LABEL")
normal_data_x = preprocessing.normalize(data_x)

# neural net
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=3197))
model.add(tf.keras.layers.Dense(10, activation = 'tanh'))
model.add(tf.keras.layers.Dense(10, activation = 'tanh'))
model.add(tf.keras.layers.Dense(2, activation = None)) # output (2 = has exoplanet, 1 = no exoplanet)

# compile model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# train model
model.fit(normal_data_x, data_y, epochs = 50)

# visualizing differences in flux for exoplanet vs non-exoplanet stars
# plt.plot(normal_data_x[1], 'b')
# plt.title("Flux vs Time for Star with Exoplant")
# plt.xlabel('Time')
# plt.ylabel('Flux')
# plt.show()

# plt.plot(normal_data_x[200], 'b')
# plt.title("Flux vs Time for Star with Exoplant")
# plt.xlabel('Time')
# plt.ylabel('Flux')
# plt.show()
