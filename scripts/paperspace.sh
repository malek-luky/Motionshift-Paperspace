#FIRST RUN
pip install --upgrade pip
c:\users\paperspace\appdata\local\programs\python\python39\python.exe -m pip install --upgrade pip
nainstalovat colmap
nainstalovat mashlab
nainstalovat git
nainstalvoat VS2019 (bicloud)
nainstalovat CUDA 11.6 (https://developer.nvidia.com/cuda-11-6-0-download-archive)
pip install rembg

# WITH CONDA
nainstalovat codna env pro windows (z webu)
conda update --all --yes
conda create --prefix /home/my_env python=3.9
activate /home/my_env/
install conda-forge (dont remember how rn)
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# WITHOUT CONDA
nainstalovat python 3.9
pip install ninja imageio PyOpenGL glfw gdown xatlas
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
git clone https://github.com/NVlabs/nvdiffrec.git
git clone https://github.com/malek-luky/Motionshift-PaperSpace.git
imageio_download_bin freeimage
(next step only if we dont have conda)
(https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


#RERUN PART
git pull (Motinshift-Paperspace folder)

#our own mask without CSPCascade
rembg p C:/Users/paperspace/Desktop/Motionshift-PaperSpace/fox/images C:/Users/paperspace/Desktop/Motionshift-PaperSpace/fox/masks
python C:/Users/paperspace/Desktop/Motionshift-PaperSpace/ones_and_zeros.py --base_dir C:/Users/paperspace/Desktop/Motionshift-PaperSpace/fox/masks #chose gray or rgb scale 
python C:/Users/paperspace/Desktop/Motionshift-PaperSpace/colmap2poses.py --aabb_scale 1 --mask_path masks --images images --colmap_path C:/Users/paperspace/Desktop/Colmap/COLMAP.bat C:/Users/paperspace/Desktop/Motionshift-PaperSpace/fox

#built in mask
python C:/Users/paperspace/Desktop/Motionshift-PaperSpace/colmap2poses.py --aabb_scale 1 --mask --images images --colmap_path C:/Users/paperspace/Desktop/Colmap/COLMAP.bat C:/Users/paperspace/Desktop/Motionshift-PaperSpace/fox

# same for both
python C:/Users/paperspace/Desktop/Motionshift-PaperSpace/remove_pics.py --base_dir C:/Users/paperspace/Desktop/Motionshift-PaperSpace/fox #TODO NEFACHA
copy "C:\Users\paperspace\Desktop\Motionshift-PaperSpace\test_fox.json" "C:\Users\paperspace\Desktop\nvdiffrec\configs"
MOVE TO NVDIFFREC FOLDER
python C:/Users/paperspace/Desktop/nvdiffrec/train.py --config C:/Users/paperspace/Desktop/nvdiffrec/configs/test_fox.json --display-interval 10
