import cv2


def downsample_image(image, scale):
    resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized_image


def convert_image_to_greyscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
