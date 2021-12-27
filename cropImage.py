import cv2
from readJason import extractTrafficLight


def cropImage(image):
    """

    :param image:
    :type image
    :return:
    """
    im = cv2.imread(image.name)  # will have to add the path here if required
    listCroppedImages = []
    # countBBox = 0
    for bbox in image.listBBox:
        crImage = im[bbox.yCoord:bbox.yCoord + bbox.height, bbox.xCoord:bbox.xCoord + bbox.width]

        # cv2.imwrite(image.name+"_" + str(countBBox)+".jpg", crImage)
        # countBBox += 1
        listCroppedImages.append(crImage)
    return listCroppedImages


# images = extractTrafficLight("result1.json")
# for image in images:
#     crImages = cropImage(image)
# print("end")
