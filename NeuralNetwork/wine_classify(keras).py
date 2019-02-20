import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split


def get_data(names):
    df = pd.read_csv('./data/wine.txt', sep=',', header=None)
    df.columns = names
    return df


def split(df):
    labels = df['label'].values
    data = df.drop('label', axis=1).values

    return data, labels


## DATA PROCESSING

column_names = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
dataframe = get_data(column_names)

# shuffle data
dataframe = dataframe.sample(frac = 1)

data, label = split(dataframe)

# normalize
data = keras.utils.normalize(data, axis=1, order=2)

train_data, test_data, train_label, test_label = train_test_split(data, label, test_size = 0.5)

    
## TRAIN MODEL (KERAS)

test_label = keras.utils.to_categorical(test_label-1,3)
train_label = keras.utils.to_categorical(train_label-1,3)

## KERAS
model = keras.Sequential()
model.add(keras.layers.Dense(32, input_dim=13, 
                             kernel_regularizer=keras.regularizers.l2(0.),
                             kernel_initializer=keras.initializers.glorot_normal(seed=0),
                             activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

optimize = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimize, metrics=['accuracy'])
model.fit(train_data, train_label, batch_size=25,epochs=5000, validation_data = (test_data, test_label))
