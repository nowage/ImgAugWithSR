# ImgAugWithSR
# Sources
|name  |orignal Git Repository                |forked Repository                    |
|------|-----------------------------------   |---------------------------------    |
|imgAug|https://github.com/aleju/imgaug       |https://github.com/nowage/imgaug     |
|MZSR  |https://github.com/JWSoh/MZSR         |https://github.com/nowage/MZSR       |
|SinGan|https://github.com/tamarott/SinGAN    |https://github.com/nowage/SinGAN     |
|Yolov5|https://github.com/ultralytics/yolov5 |https://github.com/nowage/yolov5.git |

# Base Code
## YoloV4


## MZSR
```
%%bash
cd /content/drive/MyDrive/
git clone https://github.com/nowage/MZSR
cd MZSR
python3 main.py --gpu 0 --inputpath Input/g20/Set5/ --gtpath GT/Set5/ --savepath results/Set5 --kernelpath Input/g20/kernel.mat --model 0 --num 1
ls /content/drive/MyDrive/MZSR/results/Set5/01/baby.png
```

# data
[data](./data/README.md)
