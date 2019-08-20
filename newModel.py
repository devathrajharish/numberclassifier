import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)
X_train = X_train / 255.0
X_test = test / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)
Y_train = to_categorical(Y_train, num_classes=10)


# generating more images
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1)

print("Data generated")

# building a cnn

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

print("cnn built")


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("cnn compiled")


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1)

model.fit_generator(datagen.flow(X_train2, Y_train2, batch_size=32), epochs=45, steps_per_epoch=X_train2.shape[0] // 64,
                    validation_data=(X_val2, Y_val2), callbacks=[annealer], verbose=0)

model.save('newmodel.h5')

print("model saved")

scores = model.evaluate(X_val2, Y_val2, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))



# #first model
# model = Sequential()
# model.add(Dense(num_samples,input_dim=num_samples , kernel_initializer='normal', activation='relu'))
# model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train2, Y_train2, validation_data=(X_val2, Y_val2),epochs=10,  verbose=2)
# scores=model.evaluate(X_val2, Y_val2, verbose=0)
# print("NN Error: %.2f%%" % (100 - scores[1] * 100))


# #secondmodel
# model=Sequential()
# model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
# model.add(Flatten())
# model.add(Dropout(0.4))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(X_train2, Y_train2, validation_data=(X_val2, Y_val2),epochs=10,  verbose=2)
# scores=model.evaluate(X_val2, Y_val2, verbose=0)
# print("CNN Error: %.2f%%" % (100 - scores[1] * 100))