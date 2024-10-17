'''Importing necessary libraries'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''Preparing sample data'''
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32')/255.0
X_train = np.expand_dims(X_train, -1)
X_test = X_test.astype('float32')/255.0
X_test = np.expand_dims(X_test, -1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

'''Defining CNN Model'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

'''Compile Model'''
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

'''Train Model'''
train = model.fit(X_train, y_train, epochs=3, batch_size=32)

'''Plotting the results'''
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train.history['accuracy'])
plt.title('Accuracy from Mnist dataset using CNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.plot(train.history['loss'])
plt.title('Loss from Mnist dataset using CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()