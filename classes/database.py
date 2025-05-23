from sklearn import metrics
from classes.bcolors import Colors

import cv2
import numpy as np
import os, re


class Event:
    """
    Event class: Stores pre- and post-flood images and labels
    """

    def __init__(self,
                 identifier: str,
                 pre_event_VV: list,
                 pre_event_VH: list,
                 post_event_VV: list,
                 post_event_VH: list,
                 label
                 ):
        """
        Initialize event object

        :param identifier [str]: Event identifier
        :param pre_event_VV [list]: List with images of pre-flood event (VV polarisation)
        :param pre_event_VH [list]: List with images of pre-flood event (VH polarisation)
        :param post_event_VV [list]: List with images of post-flood event (VV polarisation)
        :param post_event_VH [list]: List with images of post-flood event (VH polarisation)
        :param label [image]: Image of label
        """
        
        self.identifier = identifier
        self.pre_event = [pre_event_VV, pre_event_VH]
        self.post_event = [post_event_VV, post_event_VH]
        self.label = label


class Database:
    """
    Database class
    """

    def __init__(self,
                 pre_dir_path_VV: str,
                 pre_dir_path_VH: str,
                 post_dir_path_VV: str,
                 post_dir_path_VH: str,
                 labels_dir_path: str,
                 identifier: str
                 ):
        """
        Initialize Database object

        :param pre_dir_path_VV [str]: Path to pre-flood event directory (VV polarisation)
        :param pre_dir_path_VH [str]: Path to pre-flood event directory (VH polarisation)
        :param post_dir_path_VV [str]: Path to post-flood event directory (VV polarisation)
        :param post_dir_path_VH [str]: Path to post-flood event directory (VH polarisation)
        :param labels_dir_path [str]: Path to labels
        :param identifer [str]: Identifier for DB object
        """

        # Paths to directories
        self.paths_dirs = {'VV_pre': pre_dir_path_VV, 
                           'VH_pre': pre_dir_path_VH,
                           'VV_post': post_dir_path_VV, 
                           'VH_post': post_dir_path_VH,
                           'label': labels_dir_path}
        
        # Identifier
        self.identifier = identifier

        # List containing path pairs
        self.path_pairs = []

        # Metrics
        self.f1_score = []
        self.IoU = []
        self.precision = []
        self.recall = []

        self.mean_f1_score = 0
        self.mean_IoU = 0
        self.mean_precision = 0
        self.mean_recall = 0

    def calc_mean_nnz(self, 
                      arr: list, 
                      tol: float
                      ) -> float:
        """
        Calculates mean above tolerance

        :param arr [list]: List of values
        :param tol [float]: tolerance
        """

        new_arr = []
        for item in arr:
            if item > tol:
                new_arr.append(item)

        return np.mean(new_arr)


    def print_metrics(self):
        """
        Prints performance metrics
        """

        num_chars = 5
        tol = 0.000001

        f1_str = f'F1          '
        IoU_str = f'IoU         '
        pre_str = f'Precision   '
        rec_str = f'Recall      '
        print(f'\nPerformance metrics for modified RCNN on dataset {Colors.BOLD}{self.identifier}{Colors.ENDC}\n')
        print(f'\t\t{Colors.UNDERLINE}{Colors.BOLD}{f1_str}{IoU_str}{pre_str}{rec_str}{Colors.ENDC}')
        for i, j, k, l in zip(self.f1_score, self.IoU, self.precision, self.recall):

            i_str = f'{i:.{num_chars}f}' if i > tol else f'{Colors.RED}{i:.{num_chars}f}{Colors.ENDC}'
            j_str = f'{j:.{num_chars}f}' if j > tol else f'{Colors.RED}{j:.{num_chars}f}{Colors.ENDC}'
            k_str = f'{k:.{num_chars}f}' if k > tol else f'{Colors.RED}{k:.{num_chars}f}{Colors.ENDC}'
            l_str = f'{l:.{num_chars}f}' if l > tol else f'{Colors.RED}{l:.{num_chars}f}{Colors.ENDC}'
            
            p_str = (f'{i_str}{' ' * (len(f1_str) - num_chars - 2)}' + 
                     f'{j_str}{' ' * (len(IoU_str) - num_chars - 2)}' +
                     f'{k_str}{' ' * (len(pre_str) - num_chars - 2)}' +
                     f'{l_str}{' ' * (len(rec_str) - num_chars - 2)}')
            print(f'\t\t{Colors.ITALIC}{p_str}{Colors.ENDC}')

        p_str = (f'  Mean: {np.mean(self.f1_score):.{num_chars}f}{' ' * (len(f1_str) - num_chars - 2)}' + 
                 f'{np.mean(self.IoU):.{num_chars}f}{' ' * (len(IoU_str) - num_chars - 2)}' + 
                 f'{np.mean(self.precision):.{num_chars}f}{' ' * (len(pre_str) - num_chars - 2)}' + 
                 f'{np.mean(self.recall):.{num_chars}f}{' ' * (len(rec_str) - num_chars - 2)}')
        print(f'\n\t{Colors.BOLD}{p_str}{Colors.ENDC}\n')

        p_str = (f' Mean (culled): {self.calc_mean_nnz(self.f1_score, tol):.{num_chars}f}{' ' * (len(f1_str) - num_chars - 2)}' + 
                 f'{self.calc_mean_nnz(self.IoU, tol):.{num_chars}f}{' ' * (len(IoU_str) - num_chars - 2)}' + 
                 f'{self.calc_mean_nnz(self.precision, tol):.{num_chars}f}{' ' * (len(pre_str) - num_chars - 2)}' + 
                 f'{self.calc_mean_nnz(self.recall, tol):.{num_chars}f}{' ' * (len(rec_str) - num_chars - 2)}')
        print(f'{p_str}\n')



    def calc_performance(self, 
                   prediction, 
                   label
                   ):
        """
        Calculates performance scores for prediction and label.

        :param prediction [array]: Prediction by model
        :param label [array]: Ground truth
        """

        prediction = prediction > 0
        label = label > 0

        f1 = metrics.f1_score(label, prediction, average='micro')
        IoU = metrics.jaccard_score(label, prediction, average='micro')
        precision = metrics.precision_score(label, prediction, average='micro')
        recall = metrics.recall_score(label, prediction, average='micro')

        # Suppressing divide by zero warnings
        os.system('cls' if os.name == 'nt' else 'clear')
        
        self.f1_score.append(f1)
        self.IoU.append(IoU)
        self.precision.append(precision)
        self.recall.append(recall)


    def get_all_paths(self,
                      option: str, 
                      ) -> dict:
        """
        Gets filepaths from directory

        :param option [str]: Key to dictionary containing directory path
        """

        regex = '[0-9]+_[0-9]+'
        obj = os.scandir(self.paths_dirs[option])
        paths_dict = dict()

        for entry in obj:
            if entry.is_file() and any(x in entry.name for x in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']):
                key = re.search(regex, entry.name)
                if key is not None:
                    key = key.group()
                paths_dict.update({key: self.paths_dirs[option] + '/' + entry.name})

        return paths_dict
    
    
    def create_path_pairs(self):
        """
        Creates path-pairs, with pre, post and label
        This is necessary if dataset is incomplete (missing files)
        """

        # Regex for path matching
        regex = '[0-9]+_[0-9]+'

        pre_paths_VV = self.get_all_paths(option='VV_pre')
        post_paths_VV = self.get_all_paths(option='VV_post')

        pre_paths_VH = self.get_all_paths(option='VH_pre')
        post_paths_VH = self.get_all_paths(option='VH_post')

        label_paths = self.get_all_paths(option='label')

        for key in label_paths:
            to_match = re.search(regex, label_paths[key])
            if to_match is not None:
                to_match = to_match.group()
            if (to_match in pre_paths_VV and 
                to_match in post_paths_VV and 
                to_match in pre_paths_VH and 
                to_match in post_paths_VH):
                self.path_pairs.append({'VV_pre': pre_paths_VV[to_match],
                                        'VH_pre': pre_paths_VH[to_match],
                                        'VV_post': post_paths_VV[to_match],
                                        'VH_post': post_paths_VH[to_match],
                                        'label': label_paths[to_match]})
    

    def resize(self,
               image, 
               max_width=None, 
               max_height=None
               ):
        """
        Resizes image.

        :param image [mat]: Image to resize
        :param max_width [int]: Maximum width
        :param max_height [int]: Maximum height
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
    
    
    def read_file(self,
                   path, 
                   resize_channel=False, 
                   max_width=None,
                   max_height=None
                   ):
        """
        Reads single file from path. Optional resize.

        :param path [str]: Filepath
        :param resize_channel [bool]: To resize
        :param max_width [int]: Maximum width
        :param max_height [int]: Maximum height
        """

        if resize_channel and max_width != None:
            channel = self.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), max_width, max_height)
        else:
            channel = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        return cv2.normalize(channel, None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    

    def create_event(self, 
                     idx: int,
                      resize_channel=False, 
                      max_width=None,
                      max_height=None,
                     ) -> None:
        """
        Reads files from paths and stores them in event-objects

        :param idx [int]: Index
        """

        # Regex for identifier
        regex = '[0-9]+_[0-9]+'

        identifier = re.search(regex, self.path_pairs[idx]['VV_pre']).group()

        return Event(identifier,
                     self.read_file(self.path_pairs[idx]['VV_pre'], resize_channel, max_width, max_height),
                     self.read_file(self.path_pairs[idx]['VH_pre'], resize_channel, max_width, max_height),
                     self.read_file(self.path_pairs[idx]['VV_post'], resize_channel, max_width, max_height),
                     self.read_file(self.path_pairs[idx]['VH_post'], resize_channel, max_width, max_height),
                     self.read_file(self.path_pairs[idx]['label'], resize_channel, max_width, max_height))









    