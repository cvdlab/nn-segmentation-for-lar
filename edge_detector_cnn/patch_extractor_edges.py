"""


                                ********************************************************
                                *                                                      *
                                *  realized by Cesare Catavitello                      *
                                *                                                      *
                                *  for any question email me at cesarec88@gmail.com    *
                                *                                                      *
                                ********************************************************



=============================

This code extract and catalog all patches to give to the cnn
to achieve the edge of an image through the following criteria:
0 - non edge
1 - edge

one patch is  marked as edge when passing through 2  filters(prewitt and laplacian) the  count_center function
returns a value above a certain threshold


=============================

"""

from __future__ import print_function
from random import randint
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.filters import prewitt, laplace
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.io import imread
import numpy as np
import progressbar

progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])


class Patch_extractor(object):
    def __init__(self, num_samples=None, path_to_images=None, patch_size=(23, 23), augmentation_angle=0):
        """
        load and store all necessary information to achieve the patch extraction
        :param num_samples: number of patches required
        :param path_to_images: path to the folder containing all '.png.' files
        :param patch_size: dimensions for each patch
        :param augmentation_angle: angle necessary to operate the increase of the number of patches
        """
        print('*' * 50)
        print('Starting patch extraction...')
        if path_to_images is None:
            ValueError('no destination file')
            exit(1)
        if num_samples is None:
            ValueError('specify the number of patches to extract and reload class')
            exit(1)

        self.augmentation_angle = augmentation_angle % 360
        self.images = np.array([rgb2gray(imread(path_to_images[el]).astype('float').reshape(5, 216, 160)[-2])
                                for el in range(len(path_to_images))])
        if self.augmentation_angle is not 0:
            self.augmentation_multiplier = int(360. / self.augmentation_angle)
        else:
            self.augmentation_multiplier = 1

        self.num_samples = num_samples

        self.patch_size = patch_size

    def make_training_patches(self):
        """
        Creates datas( all patches) and labels for training CNN
        :return:
        datas : patches (num_samples, 3_chan, height, wide)
        labels (num_samples,)
        """
        classes = [0, 1]
        per_class = self.num_samples / len(classes)
        patches, labels = [], []
        progress.currval = 0
        for i in progress(xrange(len(classes))):
            p, l = self._find_patches(classes[i], per_class)
            patches.append(p)
            labels.append(l)

        return np.array(patches).reshape((self.num_samples * self.augmentation_multiplier), 3, self.patch_size[0],
                                         self.patch_size[1]), \
               np.array(labels).reshape((self.num_samples * self.augmentation_multiplier))

    def _find_patches(self, class_number, per_class):
        """
        this function in dependence of the class number search for patches with the edge pattern
        :param per_class: number of patches to find
        :param class_number the class number
        :return:
        a numpy array of patches and a numpy array of labels
        """
        print()
        patches = []
        labels = np.ones(per_class * self.augmentation_multiplier) * class_number

        ten_percent_black = 0
        ten_percent_black_value = int(float(per_class) * 0.001)
        for i in range(0, per_class):
            extracted = False
            random_image = self.images[randint(0, len(self.images) - 1)]
            while np.array_equal(random_image, np.zeros(random_image.shape)):
                random_image = self.images[randint(0, len(self.images) - 1)]
            patches_from_random = np.array(extract_patches_2d(random_image, self.patch_size, per_class))
            counter = 0
            lap_trsh = 0.53
            prew_trsh = 0.15

            while not extracted:
                if counter > per_class / 2:
                    random_image = self.images[randint(0, len(self.images) - 1)]
                    patches_from_random = np.array(
                        extract_patches_2d(random_image, self.patch_size, per_class))
                    counter = 0
                patch_1 = np.array(patches_from_random[randint(0, per_class - 1)].astype(float))
                patch = patch_1 / (256 * 256)
                edges_2 = prewitt(patch)
                edges_5 = laplace(patch)
                edges_5_n = (edges_5 + 1.) / 2

                if class_number == 1:
                    first_cond = not np.array_equal(patch, np.zeros(patch.shape))
                    if first_cond:
                        second_cond = (edges_5_n[len(patch) / 2, len(patch) / 2] > lap_trsh or
                                       self.count_center(edges_2, patch) > prew_trsh)
                        if second_cond:
                            final_patch = np.array([patch, edges_2, edges_5_n])
                            patches.append(final_patch)
                            print('*---> patch {} added'.format(i))
                            if self.augmentation_angle != 0:
                                for j in range(1, self.augmentation_multiplier):
                                    patches.append(self.rotate_patches(patch,
                                                                       edges_2,
                                                                       edges_5_n,
                                                                       self.augmentation_angle * j))

                                    print('*---> patch {} added with rotation of {} degrees '.format(i,
                                                                                                     self.augmentation_angle * j))

                            else:
                                pass
                            extracted = True

                elif class_number == 0:
                    first_cond = edges_5_n[len(patch) / 2, len(patch) / 2] <= lap_trsh and \
                                 self.count_center(edges_2, patch) <= prew_trsh
                    if first_cond:
                        if np.array_equal(patch, np.zeros(patch.shape)):
                            if ten_percent_black < ten_percent_black_value:
                                final_patch = np.array([patch, edges_2, edges_5_n])
                                patches.append(final_patch)
                                print('*---> patch {} added'.format(i))
                                ten_percent_black += 1
                                if self.augmentation_angle != 0:
                                    for j in range(1, self.augmentation_multiplier):
                                        patches.append(self.rotate_patches(patch,
                                                                           edges_2,
                                                                           edges_5_n,
                                                                           self.augmentation_angle * j))
                                        print('*---> patch {} added with rotation of {} degrees '.format(i,
                                                                                                         self.augmentation_angle * j))

                                else:
                                    pass
                                extracted = True
                            else:
                                pass
                        else:
                            final_patch = np.array([patch, edges_2, edges_5_n])
                            patches.append(final_patch)
                            print('*---> patch {} added'.format(i))
                            if self.augmentation_angle != 0:
                                for j in range(1, self.augmentation_multiplier):
                                    patches.append(self.rotate_patches(patch,
                                                                       edges_2,
                                                                       edges_5_n,
                                                                       self.augmentation_angle * j))
                                    print('*---> patch {} added'
                                          ' with rotation of {} degrees '.format(i, self.augmentation_angle * j))

                            else:
                                pass
                            extracted = True

                counter += 1

        print('extraction done for class {}'.format(class_number))
        return np.array(patches), labels

    def count_center(self, edge, patch):
        """
        this function sum the value in the square of dimension 3x3
        in the center of the patch
        :param edge: the mask obtained with the filter
        :param patch: the actual patch
        :return:
        """
        sum = 0.0
        for k in range(-1, 1):
            for j in range(-1, 1):
                sum += edge[len(patch) / 2 + k][len(patch) / 2 + j]

        return sum

    def rotate_patches(self, patch, edge_1, edge_2, rotating_angle):
        return np.array([rotate(patch, rotating_angle, resize=False),
                         rotate(edge_1, rotating_angle, resize=False),
                         rotate(edge_2, rotating_angle, resize=False)])


if __name__ == '__main__':
    pass
