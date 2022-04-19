rm -rf COCO-GAN
rm -rf ShapeNetRendering.tgz
rm -rf logs/
set -e

# create conda env
conda remove -n data_generate --all -y
conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate data_generate

pip install torch torchvision
pip install pytorch_pretrained_biggan Pillow==9.1 opencv-python gdown
git clone https://github.com/hubert0527/COCO-GAN.git

# download gan models
mkdir -p logs/CelebA_128x128_N2M2S64/ckpt/
gdown --folder https://drive.google.com/drive/folders/1WGeH4asaUeGpSmbIUJadj886z0tbXmxr --output logs/CelebA_128x128_N2M2S64/ckpt

mkdir -p logs/CelebA_128x128_N2M2S64/ckpt/
gdown --folder https://drive.google.com/drive/folders/1r-BvW6cVMHKJw-0wMI6mUepMkboWwWqN --output logs/LSUN_256x256_N2M2S128/ckpt

# generate data by GAN models
mkdir -p ../../data/LSUN_256x256_N2M2S128
mkdir -p ../../data/CelebA_128x128_N2M2S64

python COCO-GAN/main.py --config COCO-GAN/configs/LSUN_256x256_N2M2S128.yaml --test --test_output_dir ../../data/LSUN_256x256_N2M2S128
python COCO-GAN/main.py --config COCO-GAN/configs/CelebA_128x128_N2M2S64.yaml --test --test_output_dir ../../data/CelebA_128x128_N2M2S64

python generate_biggan.py
python generate_random_bg.py

# ShapNet
wget http://ftp.cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz
tar -xvf ShapeNetRendering.tgz -C ../../data
