# baseline model with dropout on the cifar10 dataset
import sys
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import os
import models
import constants
import datetime as dtime
import utils
import cv2
from data_handler import create_dataset
from sklearn.preprocessing import LabelEncoder

MAX_DATA = sys.maxsize


def load_image(filename, dir=constants.dir_cropped_Images):
    image = utils.read_image(filename, dir)
    return image


def load_tl_images(data_path="../outputs/df_data.csv",
                   test_size=0.33,
                   random_state=42,
                   from_path=constants.dir_cropped_Images):
    df_data = pd.read_csv(data_path)
    list_x = list(df_data.iloc[:, 0])
    list_y = list(df_data.iloc[:, 1])

    train_x, test_x, train_y, test_y = train_test_split(list_x,
                                                        list_y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    train_x = np.array([load_image(image_name, from_path) for image_name in train_x])
    test_x = np.array([load_image(image_name, from_path) for image_name in test_x])

    return (train_x, train_y), (test_x, test_y)


# load train and test dataset
def load_dataset():
    # load dataset
    # load ndarrays
    # (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    label_encoder = LabelEncoder()
    (train_x, train_y), (test_x, test_y) = load_tl_images()
    train_x = train_x[:MAX_DATA]
    train_y = train_y[:MAX_DATA]
    test_x = test_x[:MAX_DATA]
    test_y = test_y[:MAX_DATA]
    # one hot encode target values
    with open("../data/voc-bosch.names", "r") as fh:
        labels = [x.replace("\n", "").strip() for x in fh.readlines()]
    label_encoder.fit(labels)
    train_y = np.array(train_y)
    train_y_1 = label_encoder.transform(train_y)
    test_y = np.array(test_y)
    test_y_1 = label_encoder.transform(test_y)

    train_y = to_categorical(train_y_1, num_classes=8)
    test_y = to_categorical(test_y_1, num_classes=8)
    return train_x, train_y, test_x, test_y


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm



# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss

    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.xlabel('number of Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.legend(loc='upper left')

    # plot accuracy

    filename = f"train_history_loss.png"
    filepath = os.path.join(constants.dir_outputs, filename)
    print(filepath)
    plt.savefig(filepath, dpi=300)
    plt.close()

    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    # filename = sys.argv[0].split('/')[-1]
    plt.xlabel('number of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    filename = f"train_history_acc.png"
    filepath = os.path.join(constants.dir_outputs, filename)
    print(filepath)
    plt.savefig(filepath, dpi=300)
    # plt.show()


# run the test harness for evaluating a model
def run_test_harness(epochs=10, batch_size=32, verbose=0):
    # load dataset
    train_x, train_y, test_x, test_y = load_dataset()
    # prepare pixel data and/or data augmentation
    train_x, test_x = prep_pixels(train_x, test_x)
    # define model
    model = models.define_model_1()
    # fit model
    history = model.fit(train_x,
                        train_y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(test_x, test_y),
                        verbose=verbose)


    str_ts = dtime.datetime.now().strftime(constants.ts_fmt)
    model_path = os.path.join(constants.dir_models, f"model_{str_ts}.h5")
    model.save(model_path)

    # evaluate model
    _, acc = model.evaluate(test_x, test_y, verbose=2)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)


def predict(model, test):
    pass


def main():
    # # create dataset
    # savePath = os.path.join(constants.dir_outputs, "cropped_Images")
    # if not os.path.isdir(savePath):
    #     os.makedirs(savePath)
    # create_dataset(savePath)

    # entry point, run the test harness
    run_test_harness(epochs=50, batch_size=64, verbose=2)

    # run_test_harness(epochs=90, batch_size=1)


if __name__ == '__main__':
    main()
