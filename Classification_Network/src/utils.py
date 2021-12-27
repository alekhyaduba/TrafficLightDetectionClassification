import os
import sys
import yaml
from pandas import DataFrame as df
import cv2

import constants


class Bbox:
    def __init__(self, className, xCoord, yCoord, width, height):
        self.className = className
        self.xCoord = xCoord
        self.yCoord = yCoord
        self.width = width
        self.height = height


class Image:
    def __init__(self, name, bbox):
        """

        :param name:
        :param bbox:
        :type [bbox]
        """
        self.name = name
        self.listBBox = []
        for b in bbox:
            self.listBBox.append(b)


def convertYAML2df(filepath):
    images = yaml.safe_load(open(filepath, 'rb').read())
    dictImages = {}
    for image in images:
        boxes = image['boxes']
        imPath = image['path']
        imName = imPath.split('/')[-1]
        listBbox = []
        for box in boxes:
            x_min = box['x_min']
            y_min = box['y_min']
            width = box['x_max'] - box['x_min']
            height = box['y_max'] - box['y_min']
            label = box['label']
            bBox = Bbox(className=label, xCoord=x_min, yCoord=y_min, width=width, height=height)
            listBbox.append(bBox)
        dictImages[imName] = listBbox

    return df.from_dict(dictImages, orient='index')


def calculateIoU(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    y2 = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou * 100


def read_image(filename, dir=constants.dir_cropped_Images):
    filename = os.path.basename(filename)
    image_path = os.path.join(dir, filename)
    im = cv2.imread(image_path)
    return im


def cropImage(image, dir, savePath, shape=(64, 64)):
    """

    :param savePath:
    :param image:
    :type image
    :return:
    """
    im = read_image(image.name, dir)  # will have to add the path here if required
    dictCroppedImages = {}
    countBBox = 0
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    for bbox in image.listBBox:
        crImage = im[max(0, bbox.yCoord):max(0, bbox.yCoord) + bbox.height,
                  max(0, bbox.xCoord):max(0, bbox.xCoord) + bbox.width]
        crImage = cv2.resize(crImage, shape, interpolation=cv2.INTER_AREA)
        imageName = image.name + "_" + str(countBBox) + ".png"
        # Save the cropped image at the path given

        cv2.imwrite(os.path.join(savePath, imageName), crImage)
        countBBox += 1
        dictCroppedImages[imageName] = [bbox.xCoord, bbox.yCoord, bbox.width, bbox.height]
    return dictCroppedImages


def createLabelDf(dictCroppedImages, dfYAML):
    croppedImageLabelMap = {}
    for cImage, bBox in dictCroppedImages.items():
        parentImageName = cImage.split('_')[0]
        boxes = dfYAML.loc[parentImageName]
        max_iou = -1
        label = ''
        for box in boxes:
            if box != None:
                boxB = [box.xCoord, box.yCoord, box.width, box.height]
                iou = calculateIoU(bBox, boxB)
                if iou > max_iou:
                    label = box.className
                    max_iou = iou
        if label in ['RedLeft','Red','RedRight','GreenLeft','Green','GreenRight','Yellow','off']:
            croppedImageLabelMap[cImage] = label

    croppedImageLabelMapDf = df.from_dict(croppedImageLabelMap, orient='index')

    return croppedImageLabelMapDf
# x = convertYAML2df("train.yaml")
# print(x[0:10])
# print("end")
