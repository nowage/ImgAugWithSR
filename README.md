# ImgAugWithSR
# Sources
|name  |orignal Git Repository                |forked Repository                    |
|------|-----------------------------------   |---------------------------------    |
|imgAug|https://github.com/aleju/imgaug       |https://github.com/nowage/imgaug     |
|SinGan|https://github.com/tamarott/SinGAN    |https://github.com/nowage/SinGAN     |
|Yolov5|https://github.com/ultralytics/yolov5 |https://github.com/nowage/yolov5.git |

# Base Code
## SuperResolution
* https://github.com/ChaofWang/Awesome-Super-Resolution#2021

# imgaug

# SinGAN
### 1. Connect Remote
```
sg
```
### 2. Gpu Server Job
```
workon t1.7
pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0
cd ~/df
git clone https://github.com/nowage/SinGAN
cd SinGan
pip uninstall -y imgaug
python -m pip install -r requirements.txt
```

### 3. Train
```
rm -rf TrainedModels/*
python main_train.py --input_name birds.png

```

# yolov5

## Usage by Gpu Server
### 1. Connect Remote
```
#tm
sg
tma
```

### 2. Gpu Server Job
```
#tm
tma

docker run                        \
  --ipc=host                      \
  --gpus all                      \
  -it                             \
  -v ~/df:/usr/src/app/df         \
  --rm                            \
  --name y5                       \
  ultralytics/yolov5:latest
```

### 3. In Nvidia Docker Job
#### Setting
```
# Setting
cd ~/df/
git clone https://github.com/nowage/yolov5

cd ~/df/yolov5
pip install -r requirements.txt
pip install wandb
```

#### Data Set Download
```
dName=yolov5_mask
mkdir ~/df/data/$dName
cd ~/df/data/$dName
curl -L https://public.roboflow.com/ds/eL4QUdkpSR?key=0ikL5WLM1w > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
cat data.yaml |sed "s|\.\.|$(pwd)|" > data.yaml
```

#### Clean Old execution
```
cd ~/df/yolov5
rm -rf runs/detect/*
rm -rf runs/train/*
```

#### Training
```
dName=yolov5_mask
cd ~/df/yolov5
python train.py                                 \
  --img 416                                     \
  --batch 16                                    \
  --epochs 50                                   \
  --data /usr/src/app/df/data/$dName/data.yaml  \
  --cfg ./models/yolov5s.yaml                   \
  --weights yolov5s.pt                          \
  --name $dName
```

#### Implementation
```
cd ~/df/yolov5

tPath=/usr/src/app/df/data/$dName/test/images/
aPath=$tPath$(ls $tPath|head -1)

python detect.py                                                   \
--weights /usr/src/app/df/yolov5/runs/train/$dName/weights/best.pt \
--img 416                                                          \
--conf 0.5                                                         \
--source $aPath

ls runs/detect/exp
```

# data
[data](./data/README.md)
