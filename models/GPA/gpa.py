import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import time
import sys
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from tensorflow.keras.layers import AveragePooling2D
import cv2
from utils import load_and_preprocess_image, get_location_filter, K_tf


class GPA:
    def __init__(self, G, p, q, train_list, second_smooth=True, 
                 gpa_matrix=None, grid_point=None,
                 h=None, h_star=None, filter_size=3):
        """
        Initialization.
        
        :param G: the number of (value domain) grid points.
        :param p: image height.
        :param q: image width.
        :param train_list: list of file paths to training images (supports .png / .npy).
        :param second_smooth: (1) True: smooth the GPA-CD over pixel locations s to obtain GPA-DS;
                              (2) False: keep the GPA-CD.
        :param gpa_matrix: (1) None: compute the GPA matrix from `train_list` and `grid_point`;
                           (2) or a pre-computed GPA matrix to load directly.
        :param grid_point: (1) None: randomly generate G scalar ticks and broadcast to [G, p, q];
                           (2) or a pre-specified grid point tensor with shape [G, p, q].
        :param h: bandwidth for CD & DS
        :param h_star: bandwidth for GPA
        :param filter_size: filter size for the doubly smoothing over spatial domain (used if `second_smooth=True`).
        """
        # -- Basic inputs
        self.G = G                      # number of grid points over value domain
        self.p = p                      # image height
        self.q = q                      # image width
        self.train_list = train_list    # training image paths
        self.N0 = len(train_list)       # number of training images
        self.filter_size = filter_size  # filter size for doubly smoothing

        # -- Bandwidth
        self.bandwidth = h              # bandwidth for CD & DS
        self.bandwidth_star = h_star    # bandwidth for GPA

        # -- Construct grid point x_g^* over value domain {x_g^*: 1 <= g <= G}
        if grid_point is None:
            rng = np.random.default_rng(seed=0)  # fixed RNG for reproducibility when grid_point is None
            tick_list = rng.random(size=G)       # G grid points in (0, 1)
            # -- Broadcast each scalar tick to [p, q] and stack to [G, p, q]
            self.grid_point = tf.concat([tf.ones([1, p, q]) * tick for tick in tick_list], axis=0)
        else:
            self.grid_point = grid_point         # use user-provided grid points with [G, p, q]
        
        # -- Build or load GPA matrix
        # -- If no precomputed matrix, compute GPA-CD first; if `second_smooth=True`, spatially smooth to get GPA-DS.
        if gpa_matrix is None:
            self.gpa_matrix, self.train_time = self.compute_GPA_matrix(second_smooth=second_smooth)
            # self.gpa_matrix /= tf.reduce_max(self.gpa_matrix)
        # -- Load existing GPA matrix
        else:
            self.gpa_matrix, self.train_time = gpa_matrix, None  # skip timing when loading precomputed matrix

    def compute_GPA_matrix(self, second_smooth=True):
        """
        Compute the GPA matrix.
        :param second_smooth: (1) False: compute GPA-CD;
                              (2) True: spatially smooth to get GPA-DS.
        :return: 
        gpa_matrix: tf.Tensor, the GPA matrix with shape [G, p, q] (GPA-CD if `second_smooth=False`, else GPA-DS).
        train_time: float, wall-clock time (in seconds) for building `gpa_matrix`.
    """
        # == Stage-1: accumulate GPA-CD over training images ==
        # -- Allocate accumulator: one slice per grid tick g (G slices), each of shape [p, q]
        gpa_matrix = tf.zeros([self.G, self.p, self.q])
        t1 = time.time()
        for i in range(self.N0):
            img_path = self.train_list[i]
            
            # -- Load image as float tensor of shape [p, q] (supports .npy / image files)
            if img_path.endswith(".npy"):
                img = tf.constant(np.load(img_path))
            else:
                img = load_and_preprocess_image(img_path, (self.p, self.q))
            
            # -- Gaussian kernel over value axis:
            # -- tmp_tensor[g, s] ∝ K_h( x_g^*(s) - I_i(s) ) for every grid slice g and pixel s
            # -- Broadcasting: (img) ~ [p,q], (self.grid_point) ~ [G,p,q] → result ~ [G,p,q]
            tmp_tensor = 1 / tf.sqrt(2 * np.pi) * tf.exp(-(img - self.grid_point) ** 2 / (2 * self.bandwidth ** 2))
            tmp_tensor = tmp_tensor / (self.N0 * self.bandwidth)  # Normalize by N0 and bandwidth: standard KDE scaling
            gpa_matrix += tmp_tensor  # Accumulate contribution from the i-th training image

        # == Stage-2: optional doubly smoothing over spatial domain (GPA-DS) ==
        if second_smooth:
            # -- Build spatial filter (e.g., Gaussian-like window) for the given bandwidth & window size
            location_filter = get_location_filter(self.p, self.q, self.bandwidth, self.filter_size)
            
            # -- Depthwise conv expects shape [N, H, W, C]; we treat G as batch N and use a single channel
            Omega1 = tf.nn.depthwise_conv2d(tf.reshape(gpa_matrix, [self.G, self.p, self.q, 1]), 
                                            location_filter, strides=[1, 1, 1, 1], padding='SAME')
            
            # -- Normalize by the filter mass so the smoother preserves scale
            Omega2 = tf.reduce_sum(location_filter)
            gpa_matrix = Omega1 / Omega2
        
        gpa_matrix = tf.squeeze(gpa_matrix)  # Remove trailing singleton dims if any (keeps shape [G, p, q])
        t2 = time.time()
        train_time = t2 - t1
        return gpa_matrix, train_time

    def compute_density(self, new_img):
        """
        Compute the density of a new image.

        :param new_img: a new image tensor broadcastable to shape [p, q].
        :return:
            GPA_density: tf.Tensor, estimated density map with shape [p, q],
                         obtained by weighting the precomputed GPA matrix using bandwidth h_star.
        """
        Omega2_star = K_tf(self.grid_point - new_img, self.bandwidth_star)
        Omega1_star = Omega2_star * self.gpa_matrix
        Omega1_star = tf.reduce_sum(Omega1_star, axis=0)
        Omega2_star = tf.reduce_sum(Omega2_star, axis=0)
        GPA_density = Omega1_star / Omega2_star
        return GPA_density

    def obtain_mask(self, GPA_density, density_thres=1.75, blur_len=4,
                    blur_thres=0.15, area_thres=150, return_box=False, debug=False):
        """
        Post-process a GPA density map into a binary mask or a bounding box.
        
        :param GPA_density: per-pixel GPA density map, tensor/ndarray broadcastable to shape [p, q].
        :param density_thres: (1) float: gate densities by keeping values ≤ threshold and clamping > threshold to 1;
                              (2) None : skip gating and use GPA_density directly. Default: 1.75.
        :param blur_len: window size for AveragePooling2D (box blur) used for spatial smoothing. Default: 4.
        :param blur_thres: threshold on the smoothed map; pixels with value < blur_thres are set to 1 (foreground). Default: 0.15.
        :param area_thres: minimum connected-component area to keep; smaller components are removed as noise. Default: 150.
        :param return_box: (1) False: return a binary segmentation mask of shape [p, q];
                           (2) True : return the largest qualifying component’s bounding box (x, y, w, h, area). Default: False.
        :param debug: if True, visualize intermediate tensors (density, blurred map, mask) for inspection. Default: False.

        :return:
            If return_box is False:
                mask: np.ndarray, shape [p, q], dtype float, values in {0.0, 1.0}.
            If return_box is True:
                box_stats: np.ndarray with 5 elements (x, y, w, h, area) for the selected component,
                           or None if no component meets area_thres.
        """
        # -- [Debug] show raw GPA density
        if debug:
            plt.imshow(GPA_density)
            plt.title(f"[0] GPA_density")
            plt.colorbar()
            plt.show()

        # == Step 1: optional density gating (suppress high-density background) ==
        if density_thres is not None:
            # -- Keep values <= threshold; clamp values > threshold to 1 (background-like)
            tmp_tensor = (
            tf.cast(GPA_density <= density_thres, tf.float32) * GPA_density
            + tf.cast(GPA_density > density_thres, tf.float32)
        )
        else:
            tmp_tensor = GPA_density
        if debug:
            plt.imshow(tmp_tensor)
            plt.title(f"[1] tmp_tensor - density_thres={density_thres}")
            plt.show()

        # == Step 2: spatial smoothing via average pooling (box blur) ==
        tmp_tensor = tf.reshape(tmp_tensor, (1, self.p, self.q, 1))  # [N=1, H=p, W=q, C=1]
        avg_blur_2d = AveragePooling2D(pool_size=(blur_len, blur_len), strides=1, padding='same')
        blur_tensor = tf.squeeze(avg_blur_2d(tmp_tensor))  # back to [p, q]
        if debug:
            plt.imshow(blur_tensor)
            plt.title(f"[2] blur_tensor - avgblur={blur_len}")
            plt.colorbar()
            plt.show()
        
        # == Step 3: threshold the smoothed map to obtain a binary mask ==
        # -- Foreground (lesion) = values < blur_thres
        mask = (blur_tensor.numpy() < blur_thres) * 1.0
        if debug:
            plt.imshow(mask)
            plt.title(f"[3] mask - blur_thres={blur_thres}")
            plt.show()

        # == Step 4: connected components; remove small regions ==
        mask_uint8 = mask.astype(np.uint8)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        
        # -- `stats` rows: [x, y, width, height, area]; row 0 is background
        stats = stats[np.argsort(-stats[:, 4])]  # sort by area descending
        stats = stats[1:, ]                      # drop background row
        if return_box is False:  # retuen segmentation rather than bounding box
            # -- Keep only components with area >= area_thres
            for k in range(stats.shape[0]):
                area = stats[k, 4]
                if area < area_thres:
                    x1 = stats[k, 0]
                    y1 = stats[k, 1]
                    x2 = x1 + stats[k, 2]
                    y2 = y1 + stats[k, 3]
                    mask[y1:y2, x1:x2] = 0
            return mask
        else:
            # -- Return the largest component’s bounding box that satisfies area_thres
            if debug:
                print(stats)
            stats = stats[stats[:, 4] > area_thres, ]
            try:
                return stats[0, ]
            except:
                return None
