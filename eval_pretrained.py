import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import glob
from matplotlib import gridspec
from PIL import Image
import tensorflow as tf
from tensorflow.python.platform import gfile
import argparse
import nibabel as nib
from tqdm import tqdm

# adapted from https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'import/ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'import/SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'


  def __init__(self, name):
    """Creates and loads pretrained deeplab model."""

    self.name = name
    self.config = tf.ConfigProto(allow_soft_placement = True)
    self.sess = tf.Session(config = self.config) 
    with gfile.FastGFile(name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

    self.graph = tf.get_default_graph()

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map  = batch_seg_map[0]
    return resized_image, seg_map



def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
     A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
     for channel in range(3):
       colormap[:, channel] |= ((ind >> channel) & 1) << shift
     ind >>= 3
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
     label: A 2D array with integer type, storing the segmentation label.

    Returns:
     result: A 2D array with floating type. The element of the array
       is the color indexed by the corresponding element in the input label
       to the PASCAL color map.

    Raises:
     ValueError: If label is not of rank 2 or its value is larger than color
       map maximum entry.
    """
    if label.ndim != 2:
     raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
     raise ValueError('label value too large.')

    return colormap[label]


def viz_segmentation(image, seg_map, name):
    """Visualizes input image, segmentation map and overlay view."""

    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.clf()
    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')


    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')
    plt.savefig('output/%s_seg_overlay_pretrained.png'%(name)) 
    plt.close()

def unpad(im, s):
    s0 = im.shape[0]
    s1 = im.shape[1]
    p0 = (s0 - s)//2
    p1 = (s1 - s)//2
    return im[p0:s0-p0, p1:s1-p1] 


def resolve(seg):
    #seg[seg == 11] = 2
    seg[seg == 12] = 3
    seg[seg == 13] = 3
    return seg

 
def main():
    assert not ((args.nifti is not None) and (args.img_dir is not None)), 'must specify EITHER a NIfTI input file OR a directory with 2d images'
    assert (args.nifti is not None) or (args.img_dir is not None), 'must specify EITHER a NIfTI input file OR a directory with 2d images'
    if args.nifti is not None:
        NIFTI = True
        IMG = False
    else:
        IMG = True
        NIFTI = True
   
    print("writing output segmentation images to 'output'")
    os.system('mkdir -p output') 

    # get output file name
    if args.name is None:
        if args.img_dir is not None:
            out_name = args.img_dir[: args.img_dir.rfind('/')]
            out_name = out_name[out_name.rfind('/')+1:]
        else:
            out_name = args.nifti[args.nifti.rfind('/')+1:-4]
    else:
        out_name = args.name

    # load frozen tf graph
    model = DeepLabModel('models/pretrained_model_graph.pb') 

    if IMG:     
        files = glob.glob('{img_dir}/*png'.format(img_dir=args.img_dir)) 
        assert len(files) > 0 , 'img dir must have at least one img file' 
    else:
        # load nifti file
        img = nib.load(args.nifti)
        img = img.get_fdata().transpose(0, 2, 1)
        # scale to [0, 255]
        img = 255 * img / (np.max(img) - np.min(img))
        

    segs = []
    if IMG:
        # segment PNGs
        files = sorted(files)
        for f in tqdm(files, desc='segmenting images'):
            original_im = Image.open(f) 
            resized_im, seg_map  = model.run(original_im)
            resized_im = np.array(resized_im.resize((256, 256), resample=Image.BILINEAR))
            seg_map = cv2.resize(seg_map, dsize=(256, 256), interpolation=cv2.INTER_NEAREST) 
            seg_map = resolve(seg_map) 
            segs.append(seg_map[None, ...])
            fname = f[f.rfind('/')+1:-4]
            if args.viz: viz_segmentation(resized_im, seg_map, fname)

    else:
        # segment NIfTI sections
        for i in tqdm(range(img.shape[0]), desc='segmenting NIfTI sections'):
            im = img[i]
            im = np.flipud(im)
            im = Image.fromarray(im)
            resized_im, seg_map  = model.run(im) 
            resized_im = np.array(resized_im.resize((256, 256), resample=Image.BILINEAR))
            seg_map = cv2.resize(seg_map, dsize=(256, 256), interpolation=cv2.INTER_NEAREST) 
            seg_map = resolve(seg_map) 
            segs.append(seg_map[None, ...])
            if args.viz: viz_segmentation(resized_im, seg_map, '%s_%s' %(out_name, i))

    # concatenate and output 2d segmentations into 3d volume
    seg = np.concatenate(segs)
    seg = nib.Nifti1Image(seg, affine=np.eye(4)) 
    nib.save(seg, 'output/{out_name}.nii.gz'.format(out_name=out_name)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nifti', type=str, help='MRI 3d input in NIfTI file format')
    parser.add_argument('--img_dir', type=str, help='directory with 2d MRI images in png format')
    parser.add_argument('--viz', type=int, default=0, help='option to visualize segmentations overlaid over MRIs') 
    parser.add_argument('--name', type=str, help='name for output segmentation file. if not specified, defaults to img_dir subdirectory name or input nifti filename')

    args = parser.parse_args()
    main()
