# Deeplab MRI seg

**Volumetric brain segmentation from MRI data.** This repo has the models for brain volumetric segmentation from T1 MRI scans described in "Fast Brain Volumetric Segmentation from T1 MRI Scans" (https://link.springer.com/chapter/10.1007/978-3-030-17795-9_30). 

We fine-tune a pretrained Deeplab model (https://github.com/tensorflow/models/tree/master/research/deeplab) on 2d sections of MRI data from the ADNI (http://adni.loni.usc.edu) and OASIS (https://www.oasis-brains.org) image collections and corresponding FreeSurfer automated segmentations (https://surfer.nmr.mgh.harvard.edu). 



**Getting segmentations**

Code tested with Python 3.7.

Install git lfs (https://git-lfs.github.com/) and clone the repo. 

```bash
$ git clone https://github.com/nanand2/deeplab_mri_seg.git
$ cd deeplab_mri_seg
$ git lfs pull
```

Download required packages
```bash
$ pip install -r requirements.txt
```


We provide scripts to get segmentations from MRI data in NIfTI file format (https://nifti.nimh.nih.gov/nifti-1) or from 2d transverse/coronal/sagittal sections in PNG format.

There are two models:
 
**Pre-trained:** deeplab model trained with FreeSurfer segmentations

**Fine-tuned:** pre-trained model fine-tuned on a small set of expert annotations from the MRBrainS18 segmentation challenge (https://mrbrains18.isi.uu.nl/)


To run the pre-trained or fine-tuned model on your inputs run one of

```bash
$ python eval_pretrained.py --nifti [NIfTI input] --viz [0/1]
$ python eval_finetuned.py --img_dir [PNG directory] --viz [0/1] --name [output name]
```

Note that if you include NIfTI inputs, the script defaults to iterating over the first dimension and segmenting the 2d images corresponding to the latter dimensions.

If you point the script to an image directory, it will assume that the sorted names of the PNG files corresponds to the order of slices. The script will append the 2d segmentations into a single segmentation file in NIfTI format. 

Include the option --viz 1 to dump images (see above) of the segmented images to an output subdirectory in your current directory. The scripts handle the necessary normalization/re-scaling of the inputs.  


Segmentation classes -- pretrained model

    * 0: Background
    * 1: Cortical gray matter
    * 2: Basal ganglia
    * 3: White matter
    * 4: Cerebrospinal fluid
    * 5: Ventricles
    * 6: Cerebellum
    * 7: Brain stem
    * 8: Thalamus
    * 9: Hypothalamus
    * 10: Corpus callosum

Segmentation classes -- finetuned model

    * 0: Background
    * 1: Cortical gray matter
    * 2: Basal ganglia
    * 3: White matter
    * 4: White matter lesions
    * 5: Cerebrospinal fluid
    * 6: Ventricles
    * 7: Cerebellum
    * 8: Brain stem


Example pre-trained results
![example_pretrained results](https://github.com/nanand2/deeplab_mri_seg/blob/master/imgs/example_pretrained.png)


Example fine-tuned results
![example_finetuned results](https://github.com/nanand2/deeplab_mri_seg/blob/master/imgs/example_finetuned.png)

