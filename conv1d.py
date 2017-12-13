from keras import Sequential, Input
from keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime

import common
import numpy as np
import keras
import pandas as pd
import auc

training_end = 21
train, ans = common.load_data(range(1, training_end))
del ans['user_id']


def reshape(data):
    return np.reshape(data.values, (data.shape[0], data.shape[1], 1))


def binary(data):
    return data.applymap(lambda x: 0 if x == 0 else 1)


scaler = preprocessing.StandardScaler().fit(train)
# X = reshape(scaler.transform(train))
X = reshape(binary(train))
# X = reshape(train)

model = Sequential()
model.add(Conv1D(
    filters=10,
    # strides=7,
    kernel_size=3,
    activation='relu',
    padding='same',
    input_shape=(900, 1)
))
# model.add(BatchNormalization())

model.add(Conv1D(10, 3, padding='same',activation='relu'))
# model.add(Dropout(0.25))
# model.add(BatchNormalization())

model.add(Conv1D(10, 3, padding='same', activation='relu'))
# model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(
    280,
    activation='relu'))
# model.add(BatchNormalization())

model.add(Dense(
    7 * 4,
    activation='sigmoid'))

optimizer = keras.optimizers.adam(
    lr=0.01,
    # decay=0.99
)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy'
)

model.summary()


def schedule(epoch):
    if epoch < 60:
        lr = 1 / pow(10, epoch / 10 + 1)
        print(lr)
        return lr
    else:
        return 0.000001


hist = model.fit(
    X, ans.values,
    batch_size=100,
    epochs=1000,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.TensorBoard(histogram_freq=1),
        keras.callbacks.EarlyStopping(
            patience=6,
            verbose=1),
        # keras.callbacks.LearningRateScheduler(schedule=schedule)
    ],
    verbose=1
)


def conv1d_auc(i):
    pred_data = pd.read_csv('../con/%03d.csv' % i, index_col='user_id')
    # pred_data = scaler.transform(pred_data)
    pred_data = binary(pred_data)

    pred = model.predict(reshape(pred_data))
    pred_df = pd.DataFrame(pred)
    pred_df['auc'] = 0.0
    ans = pd.read_csv('../public/label-%03d.csv' % i, index_col='user_id')
    pi = 0
    for index, ans_row in ans.iterrows():
        a = auc.auc(pred[pi], ans_row)
        pred_df.at[pi, 'auc'] = a
        pi += 1

    cols = pred_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    pred_df = pred_df[cols]
    pred_df.to_csv(
        '../conv1d-auc/%s-%03d.csv' % (
            datetime.now().isoformat(timespec='minutes'),
            i))


for i in range(training_end - 5, training_end + 5):
    conv1d_auc(i)
