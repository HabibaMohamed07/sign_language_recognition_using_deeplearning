import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Determine the maximum length of samples
max_length = max(len(sample) for sample in data)

# Pad each sample to the maximum length
data_padded = pad_sequences(data, maxlen=max_length, padding='post', dtype='float32')

# Reshape the data to be suitable for a Convolutional Neural Network
data_padded = data_padded.reshape(data_padded.shape[0], max_length, 1)

x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Convert labels to categorical format
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 1), activation='relu', input_shape=(max_length, 1, 1)))

model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(64, kernel_size=(3, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(64, kernel_size=(3, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(len(np.unique(labels)), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
print(model.summary())


batch_size = 64
epochs = 25
training = model.fit(x_train, y_train_categorical, validation_data=(x_test, y_test_categorical),
                      epochs=epochs, batch_size=batch_size)