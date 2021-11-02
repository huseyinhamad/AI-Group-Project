import requests
import sys
import os
import getopt
import numpy as np
import matplotlib.pyplot as plt


from keras.models import load_model
from keras.preprocessing.image import load_img,  img_to_array
from numpy import expand_dims
import cv2
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


def main(argv):
    IMAGE = ""
    MODE = ""
    # Get command line arguements
    try:
        opts, args = getopt.getopt(argv, "hf:u:", ["file=", "url="])
    except getopt.GetoptError:
        print(f"USAGE: test.py -f /path/to/image  or -u /image/url")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(f"USAGE: test.py -f /path/to/image/  or -u /image/url/")
            sys.exit()
        elif opt == "-f":
            IMAGE = arg
            MODE = "file"
        elif opt == "-u":
            IMAGE = arg
            MODE = "url"
        else:
            print("No image found")
    return MODE, IMAGE


# Function to prepare image
def prepareImage(image, mode):
    imageData = ""
    if mode == "url":
        response = requests.get(image)
        with open("image.jpg", "wb") as handler:
            if not response.ok:
                print("Could not find image")
                sys.exit(41)
            else:
                for block in response.iter_content(1024):
                    if not block:
                        break
                    handler.write(block)
        # proceed to load downloaded image
        imageData = load_img("image.jpg", target_size=(256, 256))
        imageData = img_to_array(imageData)
        imageData = cv2.normalize(
            imageData, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        imageData = expand_dims(imageData, 0)

    elif mode == "file":
        imageData = load_img(image, target_size=(256, 256))
        imageData = img_to_array(imageData)
        imageData = cv2.normalize(
            imageData, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        imageData = expand_dims(imageData, 0)

    return imageData


def transformImage(image):
    # Custom object for keras load model
    cust = {'InstanceNormalization': InstanceNormalization}
    modelName = "generatorModelBtoA"
    modelNameSuffix = "h5"
    path = os.path.join(os.getcwd(), modelName+"."+modelNameSuffix)
    print(path)
    model = load_model(
        path, cust)
    predictedImage = model.predict(image)
    return image, predictedImage


def plot(image, predictedImage):
    fig = plt.figure(figsize=(3, 3))
    fig.add_subplot(2, 2, 1)

    plt.imshow(image[0].squeeze())
    plt.axis('off')
    plt.title("Actual Image")
    fig.add_subplot(2, 2, 3)
    plt.imshow(predictedImage[0].squeeze())
    plt.axis('off')
    plt.title("Generated Art")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    mode, image = main(sys.argv[1:])
    imageData = prepareImage(image, mode)
    print(imageData.shape)
    image, transformedImageData = transformImage(imageData)
    plot(imageData, transformedImageData)
