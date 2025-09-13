cd ..
git clone https://github.com/565353780/octree-shape.git
git clone https://github.com/565353780/base-trainer.git
git clone https://github.com/565353780/dino-v2-detect.git

cd octree-shape
./dev_setup.sh

cd ../base-trainer
./dev_setup.sh

cd ../dino-v2-detect
./dev_setup.sh

pip install -U timm einops diffusers flow_matching thop torchcfm \
  tos crcmod

pip install -U cupy-cuda12x

pip install flash-attn==2.8.2 --no-build-isolation
