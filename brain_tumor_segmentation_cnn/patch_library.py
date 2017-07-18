# coding=utf-8
from __future__ import print_function
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.transform import rotate
import numpy as np
import random
import os
import progressbar

progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
np.random.seed(5)


class PatchLibrary(object):
    """
    class for creating patches and subpatches from training data to use as input for segmentation models.
    """

    def __init__(self, patch_size, train_data, num_samples, augmentation_angle):
        """

        :param patch_size: tuple, size (in voxels) of patches to extract. Use (33,33) for sequential model
        :param train_data: list of filepaths to all training data saved as pngs. images should have shape (5, 216, 160)
        :param num_samples: the number of patches to collect from training data.
        :param augmentation_angle: the angle used for flipping patches(producing more datas)
        """
        # qui voglio un boolean data augmentation
        # e un valore in gradi dell'angolo
        self.patch_size = patch_size
        if augmentation_angle % 360 != 0:
            self.augmentation_multiplier = int(float(360) / float(augmentation_angle))
        else:
            self.augmentation_multiplier = 1

        self.num_samples = num_samples * self.augmentation_multiplier
        self.augmentation_angle = augmentation_angle % 360

        self.train_data = train_data
        self.h = self.patch_size[0]
        self.w = self.patch_size[1]

    def find_patches(self, class_num, num_patches):
        """
        Helper function for sampling slices with evenly distributed classes
        :param class_num: class to sample from choice of {0, 1, 2, 3, 4}.
        :param num_patches: number of patches to extract
        :return: num_samples patches from class 'class_num' randomly selected.
        """
        h, w = self.patch_size[0], self.patch_size[1]
        # qui voglio un if su data_augmentation
        patches, labels = [], np.full(num_patches, class_num, 'float')
        print('Finding patches of class {}...'.format(class_num))

        ct = 0
        while ct < num_patches:
            im_path = random.choice(self.train_data)
            fn = os.path.basename(im_path)
            try:
                label = np.array(io.imread('Labels/' + fn[:-4] + 'L.png'))
            except:
                continue

            # resample if class_num not in selected slice
            unique, counts = np.unique(label, return_counts=True)
            labels_unique = dict(zip(unique, counts))
            try:
                if labels_unique[class_num] < 10:
                    continue
            except:
                continue

            # select centerpix (p) and patch (p_ix)
            img = io.imread(im_path).reshape(5, 216, 160)[:-1].astype('float')
            p = random.choice(np.argwhere(label == class_num))
            p_ix = (p[0] - (h / 2), p[0] + ((h + 1) / 2), p[1] - (w / 2), p[1] + ((w + 1) / 2))
            patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])

            # resample it patch is empty or too close to edge
            if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (2 * h * w):
                if class_num == 0 and patch.shape == (4, h, w):
                    pass
                else:
                    continue
            patches.append(patch)
            for i in range(1, self.augmentation_multiplier):
                patch = rotate(patch, self.augmentation_angle*i, resize=False)
                patches.append(patch)
            ct += self.augmentation_multiplier

        return np.array(patches), labels

    # def slice_to_patches(self, filename):
    #     '''
    #     Converts an image to a list of patches with a stride length of 1. Use as input for image prediction.
    #     INPUT: str 'filename': path to image to be converted to patches
    #     OUTPUT: list of patched version of imput image.
    #     '''
    #     slices = io.imread(filename).astype('float').reshape(5,240,240)[:-1]
    #     plist=[]
    #     for slice in slices:
    #         if np.max(img) != 0:
    #             img /= np.max(img)
    #         p = extract_patches_2d(img, (h,w))
    #         plist.append(p)
    #     return np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))
    #

    def make_training_patches(self, entropy=False, balanced_classes=True, classes=None):
        """
        Creates datas(X) and labels(y) for training CNN
        :param entropy: if True, half of the patches are chosen based on highest entropy area.
        defaults to False.
        :param balanced_classes:  if True, will produce an equal number of each class from the randomly chosen samples.
        :param classes: list of classes to sample from.
         Only change default if entropy is False and balanced_classes is True
        :return:
        datas : patches (num_samples, 4_chan, h, w)
        labels (num_samples,)
        """
        if classes is None:
            classes = [0, 1, 2, 3, 4]
        if entropy is False:
            if balanced_classes:
                # o metto qui le ''interferenze'' prodotte da alpha o dopo ci devo pensare
                per_class = self.num_samples / len(classes)
                patches, labels = [], []
                progress.currval = 0
                for i in progress(xrange(len(classes))):
                    p, l = self.find_patches(classes[i], per_class)
                    # set 0 <= pix intensity <= 1
                    for img_ix in xrange(len(p)):
                        for slice in xrange(len(p[img_ix])):
                            if np.max(p[img_ix][slice]) != 0:
                                p[img_ix][slice] /= np.max(p[img_ix][slice])
                    patches.append(p)
                    labels.append(l)
                return np.array(patches).reshape(self.num_samples, 4, self.h, self.w), np.array(labels).reshape(
                    self.num_samples)

        else:
            print('Patches by entropy extraction... ')
            self.patches_by_entropy(self.num_samples)


if __name__ == '__main__':
    pass
