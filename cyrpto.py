from cryptography.fernet import Fernet
from numpy.lib.function_base import average

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
import numpy as np
import matplotlib.pyplot as plt
import random
import string
import sys
import time

key = Fernet.generate_key()
fernet = Fernet(key)
one = fernet.encrypt("doodoo".encode())
key = Fernet.generate_key()
fernet = Fernet(key)
two = fernet.encrypt("doodoo".encode())
print(one)
print(two)
exit()

# Number of epochs before losses are appended
# Default = 5
EPOCH_DELAY = 50

# Number of losses appended before slopes are appended
# Default = 5
LOSS_DELAY = 50

# Number of loss slopes averaged to determine training end (i.e., avg_loss_slope > 0)
# Default = 5
SLOPE_DELAY = 5

# Max training epochs
# Default = 1000
MAX_EPOCHS = 10000

# Number of training examples generated
# Default = 10000
EXAMPLES = 10000

# Minimum length of output string
# 1
STR_LEN_MIN = 1

# Maximum length of output string
# 16
STR_LEN_MAX = 16


def str_to_int_array(s):
    return [ ord(i) for i in s ]
    
def generate_output_string(N):
    return [ random.randint(0, sys.maxunicode) for i in range(N) ]

def generate_training_data(N):
    y_train = [ generate_output_string(random.randint(STR_LEN_MIN, STR_LEN_MAX)) for i in range(N) ]
    x_train = []
    for index, y in enumerate(y_train):
        y_unicode = [ chr(i) for i in y ]
        y_string = str(len(y_unicode))
        y_encode = y_string.encode()
        y_encrypt = fernet.encrypt(y_encode)
        t = time.time()
        x_string = y_encrypt.decode()
        x = []
        for c in x_string:
            x.append([ord(c)])
        x.append([t])
        x_train.append(np.array(x))
        if len(y_train[index]) < STR_LEN_MAX:
            y_train[index] = y_train[index] + [0] * (STR_LEN_MAX - len(y_train[index]))
    
    x_train = np.array(x_train, dtype=np.float)
    y_train = np.array(y_train, dtype=np.float)

    return x_train, y_train

class CustomCallback(tf.keras.callbacks.Callback):
    losses = []
    slopes = []
    def on_epoch_end(self, epoch, logs=None):
        if epoch > EPOCH_DELAY:
            self.losses.append(logs["loss"])
            if len(self.losses) > LOSS_DELAY:
                self.losses.pop(0)
                slope, intercept = np.polyfit(range(len(self.losses)), self.losses, 1)
                self.slopes.append(slope)
                if len(self.slopes) > SLOPE_DELAY:
                    self.slopes.pop(0)
                    avg = average(self.slopes)
                    print("average(self.slopes): ", avg)
                    if avg > 0:
                        while len(self.losses) > 0:
                            self.losses.pop(0)
                        while len(self.slopes) > 0:
                            self.slopes.pop(0)
                        self.model.stop_training = True

def train_model(model, learning_rate):
    learning_rate = 1 * 10 ** -learning_rate
    print("learning_rate: ", learning_rate)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    history = None

    while True:
        try:
            history = model.fit(
                x_train, 
                y_train, 
                epochs=MAX_EPOCHS,
                batch_size=100,
                callbacks=[CustomCallback()]
            )
            break
        except KeyboardInterrupt:
            print('\n\nTraining Stopped\n\n')
            break

    o = model.predict(x_train)
    print(o[0])
    print(y_train[0])

    if history:
        plt.plot(history.history["loss"])

    return model

        
x_train, y_train = generate_training_data(EXAMPLES)
print(x_train.shape)
print(y_train.shape)

model = tf.keras.models.Sequential()
model.add(Conv1D(100, 2, activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(100, 2, activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation="relu"))

for i in range(1, 20):
    model = train_model(model, i)

plt.show()

model.save("crypto_model")