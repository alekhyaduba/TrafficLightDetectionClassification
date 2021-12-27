from readJson import extractTrafficLight

import os
import pandas as pd
import constants

import utils


# def cropImage(image, savePath):
#     """
#
#     :param savePath:
#     :param image:
#     :type image
#     :return:
#     """
#     im = cv2.imread(image.name)  # will have to add the path here if required
#     listCroppedImages = {}
#     countBBox = 0
#
#     for bbox in image.listBBox:
#         crImage = im[bbox.yCoord:bbox.yCoord + bbox.height, bbox.xCoord:bbox.xCoord + bbox.width]
#         crImage = cv2.resize(crImage, (64, 64), interpolation=cv2.INTER_AREA)
#         imageName = image.name + "_" + str(countBBox) + ".png"
#         # Save the cropped image at the path given
#         cv2.imwrite(os.path.join(savePath, imageName), crImage)
#         countBBox += 1
#         listCroppedImages[imageName] = [bbox.xCoord, bbox.yCoord, bbox.width, bbox.height]
#     return listCroppedImages


# def createLabelDf(dictCroppedImages, dfYAML):
#     croppedImageLabelMap = {}
#     for cImage, bBox in dictCroppedImages.items():
#         parentImageName = cImage.split('_')[0]
#         boxes = dfYAML.loc[parentImageName]
#         max_iou = -1
#         label = ''
#         for box in boxes:
#             if box != None:
#                 boxB = [box.xCoord, box.yCoord, box.width, box.height]
#                 iou = utils.calculateIoU(bBox, boxB)
#                 if iou > max_iou:
#                     label = box.className
#                     max_iou = iou
#         if label in ['RedLeft', 'Red', 'RedRight', 'GreenLeft', 'Green', 'GreenRight', 'Yellow', 'off']:
#             croppedImageLabelMap[cImage] = label
#
#     croppedImageLabelMapDf = pd.DataFrame.from_dict(croppedImageLabelMap, orient='index')
#
#     return croppedImageLabelMapDf


def create_dataset(savePath):
    croppedImages = {}
    # Read the json file from YOLO output
    images = extractTrafficLight("../data/result.json")
    # Read the YAML file to get data
    df_test = utils.convertYAML2df("../data/train.yaml")
    for image in images:
        crImages = utils.cropImage(image,"C:/Users/alekh/OneDrive/Documents/Robotics/Datasets/rgb/train"
                                         "/traffic_light_images", savePath)
        for crIm, box in crImages.items():
            croppedImages[crIm] = box

    df_data = utils.createLabelDf(croppedImages, df_test)
    print(df_data)

    out_path = os.path.join(constants.dir_outputs, f"df_data.csv")
    df_data.to_csv(out_path)

    # return df_data


def main():
    savePath = constants.dir_cropped_Images

    if not os.path.isdir(savePath):
        os.makedirs(savePath)

    create_dataset(savePath)


if __name__ == '__main__':
    main()
