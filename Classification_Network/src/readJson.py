import json
from utils import  Bbox, Image
import os


def extractTrafficLight(fileName):
    f = open(fileName)

    data = json.load(f)
    images = []
    for i in data:
        fileName = os.path.basename(i['filename'])
        objects = i['objects']
        listBBoxes = []
        for box in objects:
            # class_id = box['class_id']
            name = box['name']
            xCord = int(box['relative_coordinates']['center_x'])
            yCord = int(box['relative_coordinates']['center_y'])
            width = int(box['relative_coordinates']['width'])
            height = int(box['relative_coordinates']['height'])
            # confidence = box['confidence']

            bBox = Bbox( name, xCord, yCord, width, height)
            listBBoxes.append(bBox)
        im = Image(fileName, listBBoxes)
        images.append(im)

    f.close()
    return images
