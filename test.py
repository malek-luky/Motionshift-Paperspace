import torch
#print(torch.zeros(1).cuda())
print(torch.cuda.is_available())


import cv2
import torch
print(torch.cuda.is_available())
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
image = cv2.imread('test/aeroplane.jpg')
mask = cv2.imread('test/mask.png', cv2.IMREAD_GRAYSCALE)

# model_path can also be specified here
# This step takes some time to load the model
#print(torch.zeros(1).cuda())
print("start")
refiner = refine.Refiner(device='cpu') # device can also be 'cpu'
print("end")

# Fast - Global step only.
# Smaller L -> Less memory usage; faster in fast mode.
output = refiner.refine(image, mask, fast=True, L=300) #fast = False, L=900

# this line to save output
cv2.imwrite('output.png', output)

plt.imshow(output)
plt.show()