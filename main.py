from classes.bcolors import Colors
from classes.rcnn import RCNN
from classes.database import*

from functions.change_vector_analysis import CVA
from functions.auto_parameters import auto_parameters

import cv2

def initialize_test(pre_dir_path_VV: str,
                    pre_dir_path_VH: str,
                    post_dir_path_VV: str,
                    post_dir_path_VH: str,
                    labels_dir_path: str,
                    identifier: str,
                    resize = True,
                    max_w = 1000,
                    t = 10,
                    gauss_std = 6,
                    gauss_dim = 21,
                    gauss_rand = True,
                    ) -> None:
    """
    :param pre_dir_path_VV [str]: Path to pre-flood eventdirectory (VV polarisation)
    :param pre_dir_path_VH [str]: Path to pre-flood event directory (VH polarisation)
    :param post_dir_path_VV [str]: Path to post-flood event directory (VV polarisation)
    :param post_dir_path_VH [str]: Path to post-flood event directory (VH polarisation)
    :param labels_dir_path [str]: Path to labels
    :param identifer [str]: Identifier

    :param resize [bool]: If True, resize input
    :param max_w [int]:  Maximum width of input
    :param max_h [int]:  Maximum height of input
    :param t [int]: Number of timesteps (~iterations)
    :param gauss_std [int]: Standard deviation of Gaussian kernel
    :param gauss_dim [int]: Dimensions of kernel (dim X dim)
    :param gauss_rand [bool]: True = RCNN, False = SPCNN
    """

    # Initialize database
    db = Database(pre_dir_path_VV,
                    pre_dir_path_VH,
                    post_dir_path_VV,
                    post_dir_path_VH,
                    labels_dir_path,
                    identifier)
    
    # Creating path pairs
    # Necessary if missing files
    db.create_path_pairs()

    # Iterate through the list of pairs
    for i in range(len(db.path_pairs)):

        # Creating event
        event = db.create_event(i, resize_channel=resize, max_width=max_w)

        # Creating difference image (negatives only)
        difference_image, _ = CVA(event.pre_event, event.post_event)

        # Auto parameter setting (Chen's method)
        beta, alpha_theta, alpha_U, V_theta, V_U = auto_parameters(difference_image)

        # Initializing RCNN
        rcnn = RCNN(difference_image,
                    t,
                    beta=beta,
                    alpha_theta=alpha_theta,
                    alpha_U= alpha_U,
                    V_theta=V_theta,
                    V_U=V_U,
                    gaussian_dim=gauss_dim,
                    gaussian_std=gauss_std,
                    random_gaussian=gauss_rand)
        
        # Evolving network
        rcnn.time_evolution()

        # Normalising output and labels (e.g. for displaying on screen or export)
        label = cv2.normalize(event.label, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        tm = cv2.normalize(rcnn.time_matrix, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # Thresholding Time Matrix w. Otsu's method
        _, thr = cv2.threshold(tm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculating performance metrics
        db.calc_performance(thr, label)

    # Printing performance metrics
    db.print_metrics()

def EMSR763_AOI01_TEST():
    initialize_test('resources/EMSR763_AOI01/PRE_S1/VV_pre/',
                    'resources/EMSR763_AOI01/PRE_S1/VH_pre/',
                    'resources/EMSR763_AOI01/POST_S1/VV_post/',
                    'resources/EMSR763_AOI01/POST_S1/VH_post/',
                    'resources/EMSR763_AOI01/Labels',
                    'EMSR763 (Tiled)')
    
def EMSR795_AOI06_TEST():
    initialize_test('resources/EMSR795_AOI06/PRE_S1/VV_pre/',
                    'resources/EMSR795_AOI06/PRE_S1/VH_pre/',
                    'resources/EMSR795_AOI06/POST_S1/VV_post/',
                    'resources/EMSR795_AOI06/POST_S1/VH_post/',
                    'resources/EMSR795_AOI06/Labels',
                    'EMSR795 (Tiled)')
    
def EMSR763_AOI01_TEST_FULL_AOI():
    initialize_test('resources/EMSR763_AOI01_FULL/PRE_S1/VV_pre/',
                    'resources/EMSR763_AOI01_FULL/PRE_S1/VH_pre/',
                    'resources/EMSR763_AOI01_FULL/POST_S1/VV_post/',
                    'resources/EMSR763_AOI01_FULL/POST_S1/VH_post/',
                    'resources/EMSR763_AOI01_FULL/Labels',
                    'EMSR763 (Full AOI)')
    
def EMSR795_AOI06_TEST_FULL_AOI():
    initialize_test('resources/EMSR795_AOI06_FULL/PRE_S1/VV_pre/',
                    'resources/EMSR795_AOI06_FULL/PRE_S1/VH_pre/',
                    'resources/EMSR795_AOI06_FULL/POST_S1/VV_post/',
                    'resources/EMSR795_AOI06_FULL/POST_S1/VH_post/',
                    'resources/EMSR795_AOI06_FULL/Labels',
                    'EMSR795 (Full AOI)')

def main():
    while True:
        p_str = (f'\n{Colors.BOLD}(1){Colors.ENDC} for EMSR763 (Tiled)' + 
                f'\n{Colors.BOLD}(2){Colors.ENDC} for EMSR795 (Tiled)' + 
                f'\n{Colors.BOLD}(3){Colors.ENDC} for EMSR763 (Full AOI)' + 
                f'\n{Colors.BOLD}(4){Colors.ENDC} for EMSR763 (Full AOI)' + 
                f'\n{Colors.BOLD}{Colors.RED}(Q/q) to quit{Colors.ENDC}\n')
        choice = input(p_str)

        match choice.lower():
            case '1':
                EMSR763_AOI01_TEST()
            case '2':
                EMSR795_AOI06_TEST()
            case '3':
                EMSR763_AOI01_TEST_FULL_AOI()
            case '4':
                EMSR795_AOI06_TEST_FULL_AOI()
            case 'q':
                return


if __name__ == '__main__':
    main()

