import argparse
import os
import sys

parser = argparse.ArgumentParser(description="convert colmap to NeRF or LLFF(NeRD) format, automask objects, and process video")
parser.add_argument('--base_dir',  type=str,
		default="", help='add the base parent directory where folder images is located')

args = parser.parse_args()
base_dir = args.base_dir

usable_files_list = []
defect_directory = os.path.join(base_dir, 'unused_images').replace('\\', '/')
defect_image_directory = os.path.join(defect_directory, 'images').replace('\\', '/')
defect_mask_directory = os.path.join(defect_directory, 'masks').replace('\\', '/')
dirs = [defect_directory, defect_image_directory, defect_mask_directory]
for d in dirs:
    if not os.path.isdir(d):
        os.makedirs(d)
with open(os.path.join(base_dir, 'view_imgs.txt')) as fp:
    for line in fp:
        usable_files_list.append(line.strip())
temp_all_img = []
temp_all_mask = []
for idx, i in enumerate(all_img):
    if os.path.split(i)[1] not in usable_files_list:
        try:
            shutil.move(i, defect_image_directory)
        except shutil.Error:
            os.unlink(i)
    else:
        temp_all_img.append(i)
for idx, i in enumerate(all_mask):
    if os.path.split(i)[1] not in usable_files_list:
        try:
            shutil.move(i, defect_mask_directory)
        except shutil.Error:
            os.unlink(i)
    else:
        temp_all_mask.append(i)
all_img = temp_all_img
all_mask = temp_all_mask