import numpy as np 
import cv2

class Activation:
    """
    Activation class
    """

    def __init__(
            self,
            activation_map,
            num_activations,
            timestep,
    ):
        """
        Args:
            activation map (ndarray): Map of internal activations.
            num_activations (int): Number of activations.
            timestep (int): Timestep of activation.
            shannon_entropy (float): Activation entropy.
        """
        
        self.activation_map = activation_map
        self.num_activations = num_activations
        self.timestep = timestep


class RCNN:
    """
    Random-Coupled Neural Network
    Reference: Liu, Haoran, Xiang, Mingrong, Liu, Mingzhe, Li, Peng, Zuo, Xue, Jiang, Xin & Zuo, Zhuo, 
                "Random-Coupled Neural Network", Electronics (Basel) 13:21 (2024), pp. 4297-, 
                doi:10.3390/electronics13214297.

    Code for RCNN-Shift by Liu et al. available at: https://github.com/HaoranLiu507/RCNNshift/tree/main
    """

    def __init__(
        self,
        input_channel,
        t: int,
        beta: float,
        alpha_theta: float,
        alpha_U: float,
        V_theta: float,
        V_U: float,
        gaussian_dim: int = 3,
        gaussian_std: int = 1,
        random_gaussian = True,
        store_activations = False,
    ) -> None:
        """
        Initialize RCNN

        :param input_channel [array]: Input image
        :param t [int]: Number of time steps to evolve network
        :param beta [float]: Linking strength coefficient
        :param alpha_theta [float]: Exponential decay coefficient of dynamic threshold
        :param alpha_U [float]: Exponential decay coefficient of the membrane potential
        :param V_theta [float]: Amplitude coefficient of dynamic threshold
        :param V_U [float]: Amplitude coefficient of membrane potential (set to 1 by convention)
        :param gaussian_dim [int]: Specifies dimension of Gaussian kernel (dim X dim)
        :param gaussian_std [int]: Standard deviation of Gaussian kernel
        :param random_gaussian [bool]: True = RCNN, False = SPCNN
        :param store_activation [bool]: If true the activations are stored in memory
        """

        # Hyperparameters
        self.t = t
        self.beta = beta
        self.alpha_theta = alpha_theta
        self.alpha_U = alpha_U
        self.V_theta = V_theta
        self.V_U = V_U

        # Channel
        self.input_channel = input_channel
        self.shape = np.shape(input_channel)
        self.N = self.shape[0] * self.shape[1]

        # Gaussian kernel parameters
        self.gaussian_dim = gaussian_dim
        self.gaussian_std = gaussian_std
        self.random_gaussian = random_gaussian

        # Storing activations
        self.store_activations = store_activations
        self.activations = []
        self.time_matrix = None
        self.num_iter = 0


    def generate_gaussian(self) -> np.ndarray:
            """
            Generates Gaussian kernel
            """

            ax = (np.linspace(-(self.gaussian_dim - 1)
                              / 2.0, (self.gaussian_dim - 1)
                              / 2.0, self.gaussian_dim))
            
            gaussian = (np.exp(-0.5 * np.square(ax)
                               / self.gaussian_std**2))
            
            kernel = np.outer(gaussian, gaussian)

            # Center element = 0 to avoid self-feedback
            if self.gaussian_dim % 2 != 0:
                kernel[int(np.ceil(self.gaussian_dim/2)) - 1, 
                       int(np.ceil(self.gaussian_dim/2)) - 1] = 0
                
            kernel = cv2.normalize(kernel, None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
            return kernel
    
    
    def generate_inactivation_matrix(self) -> np.ndarray:
        """
        Generates random inactivation matrix
        """

        inactivation_matrix = np.random.normal(0, self.gaussian_std, (self.gaussian_dim, self.gaussian_dim))
        inactivation_matrix = cv2.normalize(inactivation_matrix, None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        return inactivation_matrix
    

    def generate_weighted_inactivation_matrix(self):
        """
        Generates weighted random inactivation matrix
        """

        G = self.generate_gaussian()
        D = self.generate_inactivation_matrix()
        M = G * (D < G)
        return M /np.max(M)
    
    
    def time_evolution(self) -> None:
        """
        Evolves network in time
        Number of timesteps specified by t
        """

        # Initializing arrays to either zeros or ones
        G = self.generate_gaussian()
        U = np.zeros(self.shape)
        theta = np.ones(self.shape)
        Y = np.ones(self.shape)
        T = np.zeros(self.shape)

        # Modified beta-matrix
        im_mean = cv2.GaussianBlur(self.input_channel, (5,5), 1)
        im_mean_max = np.max(im_mean)
        noise_pixels = np.double(self.input_channel != im_mean)
        noise_norm = cv2.normalize(noise_pixels * (self.input_channel - im_mean), None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        beta_mat = 1 - noise_norm

        # Pre-calculating EAU and EAT terms
        EAU = np.exp(-self.alpha_U) 
        EAT = np.exp(-self.alpha_theta)

        c_mean = (self.input_channel + im_mean)/2

        for i in range(self.t):

            if self.random_gaussian:
                # Generates randomized kernel
                M = self.generate_weighted_inactivation_matrix()
            else:
                M = G

            # Linking input
            L = cv2.filter2D(Y, -1, M, borderType=cv2.BORDER_ISOLATED)

            # Calculating membrane potential
            U = c_mean * (1 + self.beta * beta_mat * self.V_U * L) + EAU * U

            # Pulse generator
            Y = np.double(U > theta)

            # Updating modified dynamic threshold
            theta = np.multiply(EAT, theta) + self.V_theta * Y + np.max(theta)*(im_mean_max - im_mean)/4

            # Adding to Time Matrix
            T += np.double(U > theta) * i

            if self.store_activations:
                # Creating activation object
                A = Activation(Y, np.sum(Y), i)
                self.activations.append(A)

        self.time_matrix = T
