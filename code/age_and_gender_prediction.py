"""#### Data Preprocessing"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

fldr = "UTKFace/"

files = os.listdir(fldr)

ages = []
genders = []
images = []

for fle in files:
    print(fle.split('_')[0])
    age = int(fle.split('_')[0])
    gender = int(fle.split('_')[1])
    total = fldr + '/' + fle
    image = cv2.imread(total)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    images.append(image)

for fle in files:
    age = int(fle.split('_')[0])
    gender = int(fle.split('_')[1])
    ages.append(age)
    genders.append(gender)

plt.imshow(images[24])

print(ages[24])
print(genders[24])

plt.imshow(images[53])

print(ages[53])
print(genders[53])

images_f = np.array(images)
genders_f = np.array(genders)
ages_f = np.array(ages)

np.save(fldr + 'image.npy', images_f)
np.save(fldr + 'gender.npy', genders_f)
np.save(fldr + 'age.npy', ages_f)

"""Male = 0
Female= 1
"""

values, counts = np.unique(genders_f, return_counts=True)
print(counts)

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
gender = ['Male', 'Female']
values = [4372, 5047]
ax.bar(gender, values)
plt.show()

values, counts = np.unique(ages_f, return_counts=True)
print(counts)

val = values.tolist()
cnt = counts.tolist()

plt.plot(counts)
plt.xlabel('ages')
plt.ylabel('distribution')
plt.show()

labels = []

i = 0
while i < len(ages):
    label = [[ages[i]], [genders[i]]]
    labels.append(label)
    i += 1

images_f_2 = images_f / 255

labels_f = np.array(labels)

images_f_2.shape

X_train, X_test, Y_train, Y_test = train_test_split(images_f_2, labels_f, test_size=0.25)

Y_train[0:5]

Y_train_2 = [Y_train[:, 1], Y_train[:, 0]]
Y_test_2 = [Y_test[:, 1], Y_test[:, 0]]

Y_train_2[0][0:5]

Y_train_2[1][0:5]

"""### Model"""


def Convolution(input_tensor, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(
        input_tensor)
    x = Dropout(0.1)(x)
    x = Activation('relu')(x)

    return x


def model(input_shape):
    inputs = Input(input_shape)

    conv_1 = Convolution(inputs, 32)
    maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2 = Convolution(maxp_1, 64)
    maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3 = Convolution(maxp_2, 128)
    maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    conv_4 = Convolution(maxp_3, 256)
    maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)
    flatten = Flatten()(maxp_4)
    dense_1 = Dense(64, activation='relu')(flatten)
    dense_2 = Dense(64, activation='relu')(flatten)
    drop_1 = Dropout(0.2)(dense_1)
    drop_2 = Dropout(0.2)(dense_2)
    output_1 = Dense(1, activation="sigmoid", name='gender_out')(drop_1)
    output_2 = Dense(1, activation="relu", name='age_out')(drop_2)
    model = Model(inputs=[inputs], outputs=[output_1, output_2])
    model.compile(loss=["binary_crossentropy", "mae"], optimizer="Adam",
                  metrics=["accuracy"])

    return model


Model = model((48, 48, 3))

Model.summary()

"""#### Training"""

fle_s = 'Age_gender_detection.h5'
checkpointer = ModelCheckpoint(fle_s, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False,
                               mode='auto', save_freq='epoch')
Early_stop = tf.keras.callbacks.EarlyStopping(patience=75, monitor='val_loss', restore_best_weights=True),
callback_list = [checkpointer, Early_stop]

History = Model.fit(X_train, Y_train_2, batch_size=64, validation_data=(X_test, Y_test_2), epochs=500,
                    callbacks=[callback_list])

"""### Evaluation"""

Model.evaluate(X_test, Y_test_2)

pred = Model.predict(X_test)

pred[1]

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                    wspace=0.35)

"""### For Gender"""

plt.plot(History.history['gender_out_accuracy'])
plt.plot(History.history['val_gender_out_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                    wspace=0.35)

"""### For age"""

fig, ax = plt.subplots()
ax.scatter(Y_test_2[1], pred[1])
ax.plot([Y_test_2[1].min(), Y_test_2[1].max()], [Y_test_2[1].min(), Y_test_2[1].max()], 'k--', lw=4)
ax.set_xlabel('Actual Age')
ax.set_ylabel('Predicted Age')
plt.show()

"""#### For Gender"""

i = 0
Pred_l = []
while (i < len(pred[0])):
    Pred_l.append(int(np.round(pred[0][i])))
    i += 1

report = classification_report(Y_test_2[0], Pred_l)

print(report)

results = confusion_matrix(Y_test_2[0], Pred_l)

sns.heatmap(results, annot=True)


def test_image(ind, images_f, images_f_2, Model):
    plt.imshow(images_f[ind])
    image_test = images_f_2[ind]
    pred_1 = Model.predict(np.array([image_test]))
    gender_f = ['Male', 'Female']
    age = int(np.round(pred_1[1][0]))
    gender = int(np.round(pred_1[0][0]))
    print("Predicted Age: " + str(age))
    print("Predicted Gender: " + gender_f[gender])


test_image(57, images_f, images_f_2, Model)

test_image(137, images_f, images_f_2, Model)

test_image(502, images_f, images_f_2, Model)

test_image(24, images_f, images_f_2, Model)
