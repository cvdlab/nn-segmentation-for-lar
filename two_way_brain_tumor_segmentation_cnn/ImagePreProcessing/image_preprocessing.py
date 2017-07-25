import os
import sys
import ast
import SimpleITK.SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from medpy.io import load
import subprocess
import progressbar
from glob import glob
from skimage import io
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

np.random.seed(5) # for reproducibility
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])


class ImagePreProcessing(object):
    """
    A class for pre process brain scans for one patient
    :param: path: string, path to directory of one patient. Contains following mha files:
                  flair, t1, t1c, t2,gound truth (gt)
    :param: n4itk: boolean, to specify to use n4itk normed t1 scans (default to True)
    :param: n4itk_apply: boolean, to apply and save n4itk filter to t1 and t1c scans for given patient.            
    """

    def __init__(self, path, n4itk=False, n4itk_apply=False):
        self.path = path
        self.n4itk = n4itk
        self.n4itk_apply = n4itk_apply
        self.modes = ['flair', 't1', 't1c', 't2', 'gt']
        self.slices_by_mode, self.slices_by_slice, self.normed_slices = None, None, None
        self.read_scans()
        self.norm_slices()

    def read_scans(self):
        """
        
        goes into each modality in patient directory and loads individuals scans.
        transforms scans of same slice into strip of 5 images
        :return: slice_by_slice: 
                 slice_by_mode: 
        
        """
        print 'Loading scans...'
        slices_by_mode = np.zeros((5, 176, 216, 160))
        slice_by_slice = np.zeros((176, 5, 216, 160))
        flair = glob(self.path+'/*Flair*/*.mha')
        t2 = glob(self.path+'/*_T2*/*.mha')
        gt = glob(self.path+'/*more*/*.mha')
        t1s = glob(self.path+'/**/*T1*.mha')
        t1_n4 = glob(self.path+'/*T1*/*_n.mha')
        t1 = [scan for scan in t1s if scan not in t1_n4]
        scans = [flair[0], t1[0], t1[1], t2[0], gt[0]]  # directories to each image (5 total)
        if self.n4itk_apply:
            print '-> Applyling bias correction...'
            for t1_path in t1:
                self.n4itk_norm(t1_path)  # normalize files
            scans = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
        elif self.n4itk:
            print str(t1_n4) + ' ->t1_n4'
            scans = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
        for scan_idx, scan_el in enumerate(scans):  # read each image directory, save to self.slices
            slices_by_mode[scan_idx] = io.imread(scan_el, plugin='simpleitk').astype(float)

        for mode_ix in xrange(slices_by_mode.shape[0]):  # modes 1 thru 5
            for slice_ix in xrange(slices_by_mode.shape[1]):  # slices 1 thru 155
                slice_by_slice[slice_ix][mode_ix] = slices_by_mode[mode_ix][slice_ix]  # reshape by slice
        self.slices_by_slice = slice_by_slice
        self.slices_by_mode = slices_by_mode

    def norm_slices(self):
        """
        normalizes each slice in self.slice_by_slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel  intensities
        if n4itk == True, will apply apply bias correction to T1 and T1c images
        :return: normed_slices:
        """
        print 'Normalizing slices...'
        normed_slices = np.zeros((176, 5, 216, 160))
        for slice_ix in xrange(176):
            normed_slices[slice_ix][-1] = self.slices_by_slice[slice_ix][-1]
            for mode_ix in xrange(4):
                normed_slices[slice_ix][mode_ix] = self._normalize(self.slices_by_slice[slice_ix][mode_ix])
        print 'Done.'
        self.normed_slices = normed_slices

    def _normalize(self, passed_slice):
        """  
        
        :param slice: a single slice of any modality (excluding gt).
                      all index of modality assoc with slice: 
                      (0=flair, 1=t1, 2=t1c, 3=t2).
        :return: normalized slice
        """


        b = np.percentile(passed_slice, 99)
        t = np.percentile(passed_slice, 1)
        clipped_slice = np.clip(passed_slice, t, b)
        if np.std(clipped_slice) == 0:
            return clipped_slice
        else:
            return (clipped_slice - np.mean(clipped_slice))/ np.std(clipped_slice)

    def save_patient(self, reg_norm_n4, patient_num):
        """
        :param reg_norm_n4: string, 
        :param patient_num: int, unique identifier for each patient 'reg' for original images,
                                'norm' normalized images, 'n4' for n4 normalized images
        :return: saves png in Norm_PNG directory for normed, Training_PNG for reg
        """
        print 'Saving scans for patient {}...'.format(patient_num)
        if reg_norm_n4 == 'norm':  # saved normed slices
            for slice_ix in range(176):  # reshape to strip
                strip = self.normed_slices[slice_ix].reshape(1080, 160)
                if np.max(strip) != 0:  # set values < 1
                    strip /= np.max(strip)
                if np.min(strip) <= -1:  # set values > -1
                    strip /= abs(np.min(strip))
                # save as patient_slice.png
                io.imsave('/home/ixb3/Scrivania/Norm_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)

        elif reg_norm_n4 == 'reg':
            for slice_ix in range(176):#progress(xrange(176)):
                strip = self.slices_by_slice[slice_ix].reshape(1080, 160)
                if np.max(strip) != 0:
                    strip /= np.max(strip)
                io.imsave('/home/ixb3/Scrivania/Training_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)
        else:
            for slice_ix in range(176):  # reshape to strip
                strip = self.normed_slices[slice_ix].reshape(1080, 160)
                if np.max(strip) != 0:  # set values < 1
                    strip /= np.max(strip)
                if np.min(strip) <= -1:  # set values > -1
                    strip /= abs(np.min(strip))
                # save as patient_slice.png
                io.imsave('/home/ixb3/Scrivania/n4_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)
        name = os.path.basename(self.path)[-3:]
        print name
        print 'Saving label of patient ' + str(name)
        path = glob('/home/ixb3/Scrivania/labels_toSend/HG' + str(name) + '/**/**')
        print str(path) #+ str(path[0])
        self.save_label(path[0], patient_num)
        print 'Saved!'

    def save_label(self, label, patient_num):
        """
        Load the targets of one patient in format.mha and saves the slices in format png
        :param string label: string, filepath to the label of one patient
        :param patient_num
        """

        #slices, header = load(label)
        #slices = io.imread(label, plugin='simpleitk').astype('int')
        #slices = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label))).astype(float)
        slices = io.imread(label, plugin = 'simpleitk')
        for slice_idx, slice_el in enumerate(slices):
            io.imsave('/home/ixb3/Scrivania/Labels/{}_{}L.png'.format(patient_num, slice_idx), slice_el)

    def n4itk_norm(self, path, n_dims=3, n_iters=None):
        """
        
        :param path: string,  path to mha T1 or T1c file
        :param n_dims: int,
        :param n_iters: 
        :return:  writes n4itk normalized image to parent_dir under orig_filename_n.mha
        """
        output_fn = path[:-4] + '_n.mha'
        if n_iters is None:
            n_iters = [20, 20, 10, 5]

        n4 = N4BiasFieldCorrection(output_image=output_fn)

        # dimension of input image, input image
        n4.inputs.dimension = n_dims
        n4.inputs.input_image = path
        n4.inputs.n_iterations = n_iters
        n4.run()



def save_patient_slices(patients, type):
    """
    Save RMIs and targets of one patient
    :param patients: list, string, paths to any directories of patients to save.
                    for example- glob("Training/HGG/**")
    :param type: string, 
    :return: options = reg (non-normalized), norm (normalized, but no bias correction), 
                       n4 (bias corrected and normalized) saves strips of patient slices
                       to appropriate directory (Training_PNG/, Norm_PNG/ or n4_PNG/) as patient-num_slice-num 
    """

    for patient_num, path in enumerate(patients):
        print path
        a = ImagePreProcessing(path)
        a.save_patient(type, patient_num)
        print 'Patient ' + str(patient_num) + 'saved!'

def s3_dump(directory, bucket):
    """
    necessary to work with an amzn s3 bucket
    :param directory: string, directory containing files to save
    :param bucket: string, name of s3 bucket to dump files
    :return: 
    """
    subprocess.call('aws s3 cp' + ' ' + directory + ' ' + 's3://' + bucket + ' ' + '--recursive')


if __name__ == '__main__':
    #labels = glob('/home/ixb3/Scrivania/labels_toSend/HG**/**/**.mha')
    #print 'aoooo' + str(labels)
    # labels = glob('./0001/**/**.mha')
    #le salva completamente nere
    #save_labelsss(labels).
    # patients = glob('Training/HGG/**')
    patients = glob('/home/ixb3/Scrivania/toSend/**')
    save_patient_slices(patients, 'reg')
    save_patient_slices(patients, 'norm')
  #  save_patient_slices(patients, 'n4')
    # s3_dump('Graveyard/Training_PNG/', 'orig-training-png')

