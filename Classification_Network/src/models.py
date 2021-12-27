from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD

# example of a model defined with the functional api
from tensorflow.keras import Model
from tensorflow.keras import Input


def get_functional_model():
    # define the layers
    x_in = Input(shape=(8,))
    x = Dense(10)(x_in)
    x_out = Dense(1)(x)
    # define the model
    model = Model(inputs=x_in, outputs=x_out)


# define cnn model
def define_model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def define_model_1(input_shape=(64, 64, 3), num_classes=8, dropout_p=0.4, learning_rate=0.001):
    model = Sequential()
    model.add(Conv2D(32, (7, 7), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(num_classes, activation='softmax'))
    # compile model

    opt = SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def define_model_2(input_shape=(64, 64, 3), num_classes=8, dropout_p=0.5, learning_rate=0.001):
    model = Sequential()
    model.add(Conv2D(32, (7, 7), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(num_classes, activation='softmax'))
    # compile model

    opt = SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
