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

# Build the Deep Learning model
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

# Train the model
batch_size = 128
epochs = 10
training = model.fit(x_train, y_train_categorical, validation_data=(x_test, y_test_categorical),
                      epochs=epochs, batch_size=batch_size)



# import pickle
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# # Determine the maximum length of samples
# max_length = max(len(sample) for sample in data)

# # Pad each sample to the maximum length
# data_padded = pad_sequences(data, maxlen=max_length, padding='post', dtype='float32')

# x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly!'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()



# # from sklearn.model_selection import train_test_split
# # x_train , x_test, y_train , y_test = train_test_split(images, labels, test_size=0.3, random_state=101)

# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# # batchSize = 128
# # noOfClasses = 24
# # epochs = 10

# # x_train =x_train/255
# # x_test = x_test/255

# # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# # x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# # from tensorflow.keras import backend as k
# # from tensorflow.keras.optimizers import Adam

# # model = Sequential()
# # model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
# # model.add(MaxPooling2D(pool_size=(2,2)))

# # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2,2)))

# # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2,2)))

# # model.add(Flatten())
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.20))

# # model.add(Dense(noOfClasses, activation='softmax'))

# # model.compile(loss = 'categorical_crossentropy', optimizer=Adam() , metrics=['accuracy'] )
# # print(model.summary())
# # Training = model.fit(x_train, y_train , validation_data=(x_test, y_test), epochs=epochs, batch_size= batchSize)