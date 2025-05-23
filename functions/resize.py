import cv2

def resize(image, max_width=None, max_height=None):
    """
    Function that resizes image
    
    :param image [array]: Image to resize
    :param max_width [int]: Maximum width
    :param max_height [int]: Maximum width
    """
    
    h, w = image.shape[:2]
    aspect_ratio = w/h
    
    if max_width is None:
        new_height = int(max_height/aspect_ratio)
        resized_image = cv2.resize(image, (max_height, new_height))
    else:
        new_width = int(max_width * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, max_width))

    return resized_image
