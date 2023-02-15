import cv2
import argparse
import glob
import time
import segmentation_refinement as refine

parser = argparse.ArgumentParser(description="convert colmap to NeRF or LLFF(NeRD) format, automask objects, and process video")
parser.add_argument('--base_dir',  type=str,
		default="", help='add the base parent directory where folder woth images and masks is located')

args = parser.parse_args()
base_dir = args.base_dir
images = glob.glob(base_dir+"/images/*")
images.sort()
masks = glob.glob(base_dir+"/masks/*")
masks.sort()

for i in range(0,len(images)):
    print(masks[i])
    image = cv2.imread(images[i])
    mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('test/aeroplane.jpg')
    # mask = cv2.imread('test/aeroplane.png', cv2.IMREAD_GRAYSCALE)   
    # model_path can also be specified here
    # This step takes some time to load the model
    refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'    
    # Fast - Global step only.
    # Smaller L -> Less memory usage; faster in fast mode.
    output = refiner.refine(image, mask, fast=False, L=900)     
    # this line to save output
    cv2.imwrite(masks[i], output)    