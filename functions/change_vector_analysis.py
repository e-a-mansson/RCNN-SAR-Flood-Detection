import numpy as np
import cv2

def CVA(img_list_1: list, img_list_2: list) -> np.ndarray:
    """
    Calculates the magnitude of change given two lists of activation maps.
    
    :param img_list_1 [list]: List with pre-event images
    :param img_list_2 [list]: List with post-event images

    :ret change_magnitude [array]: Magnitude of change
    :ret change_mod [array]: Magnitude of negative changes
    """

    mag_1 = np.zeros(np.shape(img_list_1[0]))
    mag_2 = np.zeros(np.shape(img_list_1[0]))
    change_magnitude = np.zeros(np.shape(img_list_1[0]))

    for channel_1, channel_2 in zip(img_list_1, img_list_2):
        change_magnitude += np.square(channel_2 - channel_1)
        mag_1 += np.square(channel_1)
        mag_2 += np.square(channel_2)

    mag_1 = np.sqrt(mag_1)
    mag_2 = np.sqrt(mag_2)

    negative_change = cv2.normalize(np.sqrt(change_magnitude) * np.double((mag_2 - mag_1) < 0), None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    change_magnitude = cv2.normalize(np.sqrt(change_magnitude), None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    change_mod = np.float32(change_magnitude * negative_change)

    return change_mod, change_magnitude
