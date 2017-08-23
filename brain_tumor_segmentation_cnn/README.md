# CNN For HGG/LGG Brain tumor Segmentation

Is a convolutional neural network inspired on the paper of *[ (S. Pereira et al.)]( http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7426413&isnumber=7463083)*  for the model implementation and the code of [Nikki Aldeborgh
(naldeborgh7575)](https://github.com/naldeborgh7575/brain_segmentation) for the patch extraction and image preprocessing.


The model for HGG tumor modality is the following:

| Layers      | Input           | Output |
| --- |---|---|
| Convolution | 4x33x33 | 64x33x33 |
|Leaky Relu| 64x33x33 | 64x33x33 |
| Convolution  | 64x33x33| 64x33x33|
|Leaky Relu| 64x33x33 | 64x33x33 |
| Convolution  | 64x33x33| 64x33x33|
|Leaky Relu| 64x33x33 | 64x33x33 |
|Max Pooling|  64x33x33| 64x16x16 |
| Convolution  | 64x16x16| 128x16x16|
|Leaky Relu | 128x16x16| 128x16x16|
| Convolution  | 128x16x16| 128x16x16|
| Convolution  | 128x16x16| 128x16x16|
|Leaky Relu | 128x16x16| 128x16x16|
|Max Pooling| 128x16x16| 128x7x7|
|Fully Connected|6272|256|
|Fully Connected|256|5|




 For LGG tumor modality instead the following

| Layers      | Input           | Output |
| --- |---|---|
| Convolution | 4x33x33 | 64x33x33 |
|Leaky Relu| 64x33x33 | 64x33x33 |
| Convolution  | 64x33x33| 64x33x33|
|Leaky Relu| 64x33x33 | 64x33x33 |
|Max Pooling|  64x33x33| 64x16x16 |
| Convolution  | 64x16x16| 128x16x16|
|Leaky Relu| 128x16x16| 128x16x16|
| Convolution  | 128x16x16| 128x16x16|
|Max Pooling| 128x16x16| 128x7x7|
|Fully Connected|6272|256|
|Fully Connected|256|5|


For both models in the and is used the SoftMax activation function.

1.  In 
	* brain_pipeline
	* patch_extractor
	* patch_library
 the conversion of all '.mha'  files into '.png'  to all brain images is performed. To each brain image from every patient, all different modalities ( (FLAIR), T1, T1-contrasted, and T2 )   are put together into one  single stripe .  The output for an image is the following:
[image](https://github.com/cvdlab/nn-segmentation-for-lar/blob/master/brain_tumor_segmentation_cnn/readme/1.png)

 You can find brain_pipeline [here.](https://github.com/cvdlab/nn-segmentation-for-lar/tree/master/pre_processing)

2.  in Segmentaion_Model the cnn models are created and compiled. Is possible to choose between hgg or lgg model (as descibed in the article) depending from which kind of tumoral pattern is treated.

3.  in image_png_converter there is a randomic conversion of a mha file into png file in order to output some testing material.

The workflow is described as follow:
<img src="readme/Brain Tumor Segmentation Pipeline.png">


## Results 
This is the segmented image with a low number of patches 10000

<img src="readme/result0_10000.png " width="100">
<img src="readme/color_code.png " width="100">

For a sharpen result 100000 or 150000 patches are reccomended 


## How to use

### Ho to run Cnn

	user path/to/package $	python Segmentation_Model.py -option expected_value


### All available Options
	
	-train','-t',		set the number of data to train with default=1000 (int value expected)
                        
	-augmentation','-a',		set data augmentation option through rotating angle express values in degrees, as default no augmentation is made (int value expected)
	
	'-modality','-m',		set to use model for hgg(True) or lgg(False), default=True (boolean value expected)
                        
	'-load','-l',		load the model already trained, as default no load happen. insert model name as: 'model_name' (string value expected)
	
	'-save', '-s',	save the trained model in the specified path, as  default no save happen( the name and all it's specification happens automatically) (no value expected)
	
	'-test',			execute test with the expressed datas (no value expected)