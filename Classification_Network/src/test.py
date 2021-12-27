import os

import cv2
from sklearn.preprocessing import LabelEncoder

from readJson import extractTrafficLight
import utils
from keras.saving import save

from tensorflow.keras import Model

import pandas as pd
import constants
import numpy as np


# read the result.json
# for each bounding box extract the label
# read YAML and this will have original label
# use the extracted label to crop images and feed to the classifier for prediction and store the results in the another df
#
# images = extractTrafficLight("../data/result_test.json")
# df_test = utils.convertYAML2df("../data/test.yaml")
# croppedImages = {}
# for image in images:
#     crImages = utils.cropImage(image, "C:/Users/alekh/OneDrive/Documents/Robotics/Datasets/rgb/test",
#                                constants.dir_cropped_Images)
#     for crIm, box in crImages.items():
#         croppedImages[crIm] = box

def prep_pixels1(imagesAsArray):
    # convert from integers to floats
    # imageAsArray_norm = np.asarray(imagesAsArray).astype('float32')

    # normalize to range 0-1
    imageAsArray_norm = imagesAsArray / 255.0
    # return normalized images
    return imageAsArray_norm


#
# def createDictActualLabel():
#     dict_actual = {
#         "24220.png": {
#             "bbox": []
#             "labels": ["Red", "Green"]
#         },
#
#     }
#     return dict_actual
#
# def iou_based_label(list_ref_boxes, pBox):
#
#     maxIOU = -1
#     aLabel = ''
#     for index, aBox in enumerate(list_ref_boxes):
#         if aBox != None:
#             iou = utils.calculateIoU([aBox.xCoord, aBox.yCoord, aBox.width, aBox.height],
#                                      [pBox.xCoord, pBox.yCoord, pBox.width, pBox.height])
#             if iou > maxIOU:
#                 aLabel = aBox.className
#                 maxIOU = iou
#
#     return aLabel
#
# def get_actual_label(list_queries):
#     list_queries = [
#         {"filename": "24220.png", "bbox":[aBox.xCoord, aBox.yCoord, aBox.width, aBox.height]}
#     ]
#     dict_actual =

def getYoloAccuracy(resultFileName):
    correctPrediction = 0
    totalPredictions = 0
    actualBoxes = 0
    images = extractTrafficLight(resultFileName)
    df_test = utils.convertYAML2df("../data/test.yaml")
    for i in range(0, len(df_test)):
        # actualBoxes += len(df_test.iloc[i])
        boxes = df_test.iloc[i]
        for bx in boxes:
            if bx != None:
                actualBoxes += 1
    for image in images:
        predictionBBoxes = image.listBBox
        actualBBoxes = df_test.loc[image.name]
        for pBox in predictionBBoxes:
            totalPredictions += 1
            maxIOU = -1
            aLabel = ''
            for aBox in actualBBoxes:
                if aBox != None:

                    iou = utils.calculateIoU([aBox.xCoord, aBox.yCoord, aBox.width, aBox.height],
                                             [pBox.xCoord, pBox.yCoord, pBox.width, pBox.height])
                    if iou > maxIOU:
                        aLabel = aBox.className
                        maxIOU = iou
            if aLabel == pBox.className:
                correctPrediction += 1

    return correctPrediction * 100 / actualBoxes


def getClassifierAccuracy(dfPredicted):
    correctPrediction = 0
    totalPredictions = 0
    actualBoxes = 0
    # images = extractTrafficLight(resultFileName)
    images = list(dfPredicted["Image"])
    predictionBBoxes = list(dfPredicted["BBox"])
    pred_labels = list(dfPredicted["y_pred"])
    df_test = utils.convertYAML2df("../data/test.yaml")
    for i in range(0, len(df_test)):
        # actualBoxes += len(df_test.iloc[i])
        boxes = df_test.iloc[i]
        for bx in boxes:
            if bx != None:
                actualBoxes += 1
    for image, pBox, pLabel in zip(images, predictionBBoxes, pred_labels):
        imageName = image.split('_')[0]
        actualBBoxes = df_test.loc[imageName]
        maxIOU = -1
        aLabel = ''
        boxExists = False
        for aBox in actualBBoxes:
            if aBox != None:
                boxExists = True


                iou = utils.calculateIoU([aBox.xCoord, aBox.yCoord, aBox.width, aBox.height],
                                         pBox)
                if iou > maxIOU:
                    aLabel = aBox.className
                    maxIOU = iou
        if boxExists:
            totalPredictions += 1
            if aLabel == pLabel:
                correctPrediction += 1

    return correctPrediction * 100 / totalPredictions


def getPredictedTL(resultFileName):
    dictCroppedImages = {}
    images = extractTrafficLight(resultFileName)
    # Read the YAML file to get data
    # df_test = utils.convertYAML2df("../data/test.yaml")
    for image in images:
        im = utils.read_image(image.name, "C:/Users/alekh/OneDrive/Documents/Robotics/Datasets/rgb/test")

        countBBox = 0
        for bbox in image.listBBox:
            crImage = im[max(0, bbox.yCoord):max(0, bbox.yCoord) + bbox.height,
                      max(0, bbox.xCoord):max(0, bbox.xCoord) + bbox.width]
            crImage = cv2.resize(crImage, (64, 64), interpolation=cv2.INTER_AREA)
            imageName = image.name + "_" + str(countBBox) + ".png"
            # Save the cropped image at the path given

            cv2.imwrite(os.path.join(constants.dir_cropped_Images, imageName), crImage)
            countBBox += 1
            dictCroppedImages[imageName] = [imageName, [bbox.xCoord, bbox.yCoord, bbox.width, bbox.height],
                                            bbox.className]

    df_tls = pd.DataFrame.from_dict(dictCroppedImages, orient='index')

    # df.columns = ["Image", "BBox", "Prediction_YOLO"]
    out_path = os.path.join(constants.dir_outputs, f"df_data_tls.pkl")

    df_tls.head().to_pickle(f"../outputs/df_data_small.pkl")
    df_tls.to_pickle(out_path)
    #
    # for crIm, box in crImages.items():
    #     cropped_Images[crIm] = box
    # df_data = utils.createLabelDf(cropped_Images, df_test)
    # print(df_data)

    # out_path = os.path.join(constants.dir_outputs, f"df_data_test.csv")
    # df_data.to_csv(out_path)
    return dictCroppedImages


# print(f"Accuracy of correctly classified traffic signal from YOLO: ", getYoloAccuracy("../data/result_test.json"))


def multi_y_pred_2single(y_pred):
    label_encoder = LabelEncoder()
    with open("../data/voc-bosch.names", "r") as fh:
        labels = [x.replace("\n", "").strip() for x in fh.readlines()]
    label_encoder.fit(labels)
    list_pred = []
    for pred in y_pred:
        index = np.argmax(pred)
        list_pred.append(index)

    return label_encoder.inverse_transform(list_pred)


#
#
def main():
    # c = getPredictedTL("../data/result_test.json")

    # df = pd.read_pickle(f"../outputs/df_data_tls.pkl")
    # df.columns = ["Image", "BBox", "Prediction_YOLO"]
    # list_images = list(df["Image"])
    # #
    # ims = np.array([cv2.imread(f"../outputs/cropped_Images/{image_name}") for image_name in list_images])
    # norm_im = prep_pixels1(ims)
    #
    # model = save.load_model("../models/final_model.h5")
    # y_pred = model.predict(norm_im)
    #
    # # print(y_pred.shape, y_pred[0])
    # y = multi_y_pred_2single(y_pred)
    # df["y_pred"] = y
    # df.to_pickle(f"../outputs/df_data_pred.pkl")
    # print(y)

    df_pred = pd.read_pickle(f"../outputs/df_data_pred.pkl")
    print(getClassifierAccuracy(df_pred))

    print("end")


if __name__ == '__main__':
    main()
