# Import the necessary libraries
from PIL import Image
from numpy import asarray
import numpy as np
import argparse
import cv2
import glob

parser = argparse.ArgumentParser(description="convert colmap to NeRF or LLFF(NeRD) format, automask objects, and process video")
parser.add_argument('--base_dir',  type=str,
		default="", help='folder where masks pictures are located')

args = parser.parse_args()
base_dir = args.base_dir
filenames = glob.glob(base_dir+"/*")
filenames.sort()

for file in filenames:
	#GRAYSCALE
	#grey_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	#grey_img[grey_img>1]=255
	#cv2.imwrite(file,grey_img)

	#COLOR
	img = cv2.imread(file)
	img[img>0]=255
	cv2.imwrite(file,img)

#TEST1
# asarray() class is used to convert
# PIL images into NumPy arrays
#numpydata = asarray(img)
#numpydata = np.sum(numpydata, axis=2)
#numpydata[numpydata>0]=1
#print(np.sum(numpydata))

#TEST2
#im = Image.fromarray(numpydata)
#im.save("mask.png")
#np.save("mask2.png", numpydata) 