import cv2
import numpy as np

def auto_parameters(img: np.ndarray) -> float:
    """
    Automatically calculates the parameters 
    (beta, alpha_theta, alpha_U, V_theta, V_U) 
    for the RCNN.

    Reference: Yuli Chen, Sung-Kee Park, Yide Ma & Ala, R, 
                "A New Automatic Parameter Setting Method of a Simplified PCNN for Image Segmentation", 
                IEEE transactions on neural networks 22:6 (2011), pp. 880-892, 
                doi:10.1109/TNN.2011.2128880.

    :param img [array]: Array to calculate parameters for  
    """
    
    # Often set to 1 according to available literature.
    V_U = 1.0

    # Alpha U
    channel_norm = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    alpha_U = np.log(1/np.std(channel_norm))

    # Beta
    channel_norm_0_255 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    S_prime, _ = cv2.threshold(channel_norm_0_255, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    S_prime = S_prime/255
    S_max = np.max(img)
    beta = ((S_max/S_prime) - 1)/6*V_U

    # V theta
    V_theta = np.exp(-alpha_U) + 1 + 6*beta*V_U

    # Alpha theta
    M3 = (1 - np.exp(-3*alpha_U))/(1 - np.exp(-alpha_U)) + 6*beta*V_U*np.exp(-alpha_U)
    alpha_theta = np.log(V_theta/(S_prime*M3))

    return beta, alpha_theta, alpha_U, V_theta, V_U