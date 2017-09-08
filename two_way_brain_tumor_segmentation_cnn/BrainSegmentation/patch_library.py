import numpy as np
import random
import os
import h5py
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import gray2rgb
from skimage.filters.rank import entropy
from skimage.morphology import disk
import progressbar
from sklearn.feature_extraction.image import extract_patches_2d

progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
np.random.seed(5)


class PatchLibrary(object):
    def __init__(self, train_samples, num_samples, label_folder_path, patch_size=(65, 65), subpatches_33=True):
        '''
        class for creating patches and subpatches from training data to use as input for segmentation models.
        INPUT   (1) list 'train_samples': list of filepaths to all training data saved as pngs.
                        images should have shape (5, 216, 160)
                (2) int 'num_samples': the number of patches to collect from training data.
                (3) tuple 'patch_size': size (in voxels) of patches to extract. Default= (65,65)

                (4) bool 'subpatches_33': if true for every patches a subpatches 33x33 is extracted. Default=True
        '''

        self.patch_size = patch_size
        self.num_samples = num_samples
        self.train_samples = train_samples
        self.h = self.patch_size[0]
        self.w = self.patch_size[1]
        self.subpatches_33 = subpatches_33
        self.train_data = self.set_train_data(label_folder_path)

    def set_train_data(self, label_folder_path):
        '''
        Couples together the path of every RMI strip with the relative targets matrix
        :return: list 'train_data', containing the path of training_sample coupled with targets, a numpy array
            containing the labels of each pixel.
        '''
        train_data = []
        for sample_path in self.train_samples:
            name = os.path.basename(sample_path)[:-4]
            label = io.imread(label_folder_path + '/' + name + 'L.png')
            sample_label = [sample_path, label]
            train_data.append(sample_label)
        return train_data



    def find_balanced_patches(self, class_num, num_patches):
        '''
        Helper function for sampling slices with evenly distributed classes
        INPUT:  (1) int 'class_num': class to sample from choice of {0, 1, 2, 3, 4}.
                (2) int 'num_patches': number of patches to be generated
        OUTPUT: (1) num_samples patches from class 'class_num' randomly selected.
        '''
        h, w = self.patch_size[0], self.patch_size[1]
        patches, labels = [], np.full(num_patches, class_num, 'float')
        print 'Finding patches of class {}...'.format(class_num)

        ct = 0
        while ct < num_patches:
            sample_label = random.choice(self.train_data)
            # resample if class_num not in selected slice
            #problema: non trova label delle classi 2, 3, 4
            if len(np.argwhere(sample_label[1] == class_num)) < 1:
                print 'Not enough pixels with label' + str(class_num)
                continue
            # select centerpix (p) and patch (p_ix)
            img = io.imread(sample_label[0])
            img = img.reshape(5, 216, 160)[:-1].astype('float')
            p = random.choice(np.argwhere(sample_label[1] == class_num))
            p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
            patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
            # resample if patch is empty or too close to edge
            if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (h * w):
                print 'Patch empty or too close to the edge '
                continue
            patches.append(patch)
            ct += 1
            print 'Patch number ' + str(ct) + 'choosen!'
        return np.array(patches), class_num

    def find_patches(self, num_patches):
        '''
        extract randomly patches from training set
        :param num_patches: number of patches to extract
        :return: patches, labels; list of extracted patches and relative labels
        '''
        h, w = self.patch_size[0], self.patch_size[1]
        patches = []
        labels = []
        print 'Finding unbalanced patches...'

        ct = 0
        while ct < num_patches:
            sample_label = random.choice(self.train_data)
             # select centerpix (p) and patch (p_ix)
            img = io.imread(sample_label[0])
            img = img.reshape(5, 216, 160)[:-1].astype('float')
            p = random.choice(np.argwhere(sample_label[1] != -1))
            # patch 65x65 around the selected pixel
            p_ix = (p[0] - (h / 2), p[0] + ((h + 1) / 2), p[1] - (w / 2), p[1] + ((w + 1) / 2))
            patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
            # resample if patch is empty or too close to edge
            if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (h * w):
                print 'Patch empty or too close to the edge '
                continue

            patches.append(patch)
            labels.append(sample_label[1][p[0]][p[1]])
            ct += 1
            print 'Patch number ' + str(ct) + ' of label' + str(sample_label[1][p[0]][p[1]]) + 'choosen!'
        return np.array(patches), labels

    def center_n(self, n, patches):
        '''
        Takes list of patches and returns center nxn for each patch. Use as input for cascaded architectures.
        INPUT   (1) int 'n': size of center patch to take (square)
                (2) list 'patches': list of patches to take subpatch of
        OUTPUT: list of center nxn patches.
        '''
        sub_patches = []
        for mode in patches:
            subs = np.array([patch[(self.h/2) - (n/2):(self.h/2) + ((n+1)/2), (self.w/2) - (n/2):(self.w/2) + ((n+1)/2)]
                             for patch in mode])
            sub_patches.append(subs)
        return np.array(sub_patches)

    def slice_to_patches(self, filename):
        '''
        Converts an image to a list of patches with a stride length of 1. Use as input for image prediction.
        INPUT: str 'filename': path to image to be converted to patches
        OUTPUT: list of patched version of input image.
        '''
        slices = io.imread(filename).astype('float').reshape(4, 216, 160)[:-1]
        plist = []
        for img in slices:
            if np.max(img) != 0:
                img /= np.max(img)
            p = extract_patches_2d(img, (self.h, self.w))
            plist.append(p)
        return np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))

    def make_training_patches(self, balanced_classes=True, classes=[0, 1, 2, 3, 4]):
        '''
        Creates X65, X33 and y, respectively the patches 65x65 and 33x33 around the same pixel and the label of that pixel for training DCNN
        INPUT   (1) bool 'entropy': if True, half of the patches are chosen based on highest entropy area. defaults to False.
                (2) bool 'balanced classes': if True, will produce an equal number of each class from the randomly chosen samples
                (3) list 'classes': list of classes to sample from. Only change default oif entropy is False and balanced_classes is True
        OUTPUT  (1) X: patches (num_samples, 4_chan, h, w)
                (2) y: labels (num_samples,)
        '''
        if balanced_classes:
            per_class = self.num_samples / len(classes)
            x_patches = []
            labels = []
            for i in range(5):#progress(xrange(len(classes))):
                p, l = self.find_balanced_patches(classes[i], per_class)
                # set 0 <= pix intensity <= 1
                for img_ix in xrange(len(p)):
                    for slice in xrange(len(p[img_ix])):
                        if np.max(p[img_ix][slice]) != 0:
                            p[img_ix][slice] /= np.max(p[img_ix][slice])
                    x_patches.append(p[img_ix])
                    labels.append(l)
        else:
            x_patches = []
            p, labels = self.find_patches(self.num_samples)
            # set 0 <= pix intensity <= 1
            for img_ix in xrange(len(p)):
                for slice in xrange(len(p[img_ix])):
                    if np.max(p[img_ix][slice]) != 0:
                        p[img_ix][slice] /= np.max(p[img_ix][slice])
                x_patches.append(p[img_ix])
        if self.subpatches_33:
            x33_patches = self.center_n(33, x_patches)
            print '33x33 subpatches extracted!'
            return x33_patches, x_patches, labels
        else:
            print 'x= ' + str(x_patches)
            print 'lab= ' + str(labels)
            return x_patches, labels

if __name__ == '__main__':
    pass