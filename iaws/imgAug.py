# -*- coding: utf-8 -*-

"""Image Augmentation Module
# Glossary
* bb : Bounding Box
* bbs : Bounding BoxeS

"""
import imgaug.augmenters as iaa
import imageio
import imgaug as ia
import numpy as np
import pandas as pd
import os, sys
import cv2


from matplotlib.pyplot import imsave, imshow
import shutil

def imgAugResize(img,bs,longerSide=224,isImageWithBb=False):
    """
    imgAugResize
    """
    bbs = ia.BoundingBoxesOnImage(bs, shape=img.shape)
    seq = iaa.Sequential([
        iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": longerSide,"interpolation":"cubic" })
    ])
    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([img])[0]

    bbs_resized=seq_det.augment_bounding_boxes(bbs)
    # print(p,bbs_aug)


    image_aug=bbs_resized.draw_on_image(image_aug) if isImageWithBb else image_aug
    images_resized=image_aug

    return (images_resized,bbs_resized.bounding_boxes)

def bbsYolo5ToBbsImgAug(img,bbs):
    '''
    Convert Yolo5 bbs format to ImgAug bbs.
    '''
    bbsImgAug = []
    w = img.shape[1]
    h = img.shape[0]
    for r in bbs.iterrows():
        ll=r[1].to_list()
        wrh=ll[3]/2 # width ratio half
        hrh=ll[4]/2 # hight ratio half
        x1= ( ll[1]-wrh )*w
        x2= ( ll[1]+wrh )*w
        y1= ( ll[2]-hrh )*h
        y2= ( ll[2]+hrh )*h
        bbsImgAug.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2,label=str(int(ll[0]))))

    return bbsImgAug


def addBoudingBox(img,bbs):
    """
    addBoudingBox..
    """
    if type(bbs) is list:
        bs = bbs
    else:
        bs = bbsYolo5ToBbsImgAug(img,bbs)

    bbsoi = ia.BoundingBoxesOnImage(
        bs,
        shape=img.shape
    )
    imgBouded = bbsoi.draw_on_image(img,color=(255, 0, 0), size=3)
    return imgBouded



def bbsImgAugToBbsYolo5(img,bbs):
    '''
    Convert ImgAug bbs format to Yolo5 bbs.
    @param img:s
    @param bbs:s
    @return: s

    @variable c0: class
    @variable c1: 중앙에서 X축으로 이동된 비율
    @variable c2: 중앙에서 Y축으로 이동된 비율
    @variable c3: X축에서 차지하는 비율
    @variable c4: Y축이서 차지하는 비율
    '''
    bbsYolo5=[]
    w=img.shape[1] #  width
    h=img.shape[0] #  height
    ox=w/2  # origin x
    oy=h/2  # origin y
    for bb in bbs:
        c0=bb.label
        c1= ((bb.x2 + bb.x1)/2 ) / w
        c2= ((bb.y2 + bb.y1)/2) / h
        c3= abs(bb.x2 - bb.x1)/w
        c4= abs(bb.y2 - bb.y1)/h
        bbsYolo5.append([c0,c1,c2,c3,c4])
        df_bbsYolo5=pd.DataFrame(bbsYolo5)
    return df_bbsYolo5


def imgAugCropTo9(img, bs, isImageWithBb=False, size=400):
    """
    imgAugCropTo9
    """
    bbs = ia.BoundingBoxesOnImage(bs, shape=img.shape)
    images_croped = []
    bbs_croped = []
    position = ['center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-bottom', 'right-top',
                'right-center', 'right-bottom']

    for p in position:
        seq = iaa.Sequential([
            iaa.CropToFixedSize(width=size, height=size, position=p)
        ])
        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([img])[0]

        bs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        bbs_croped.append(bsJustify(image_aug,bs_aug))
        # print(p,bbs_aug)

        image_aug = bs_aug.draw_on_image(image_aug) if isImageWithBb else image_aug
        images_croped.append(image_aug)

    return (images_croped, bbs_croped)
def bsJustify(img, bs):
    '''
    칸이 넘어가는 바운딩박스를 제거하거나 자름
    @param img:
    @param bs:
    @return:
    '''
    w = img.shape[1]
    h = img.shape[0]
    bs_justified = []
    inBound = True
    for b in bs:
        inBound = True
        if b.x1 < 0: b.x1 = 0
        if b.x1 > w: inBound = False
        if b.x2 < 0: inBound = False
        if b.x2 > w: b.x2 = w
        if b.x1 == b.x2: inBound = False

        if b.y1 < 0: b.y1 = 0
        if b.y1 > h: inBound = False
        if b.y2 < 0: inBound = False
        if b.y2 > h: b.y2 = w
        if b.y1 == b.y2: inBound = False

        if inBound: bs_justified.append(b)
    return bs_justified

def resizeImageBatch(from_images_path, to_images_path, from_bbs_path, to_bbs_path, longerSide=224):
    '''
    Multi Image Resize with bbs.
    @param from_images_path:
    @param to_images_path:
    @param from_bbs_path:
    @param to_bbs_path:
    @param longerSide:
    @return:
    '''
    file_list = os.listdir(from_images_path)
    for img_name in file_list:
        #print(from_images_path + img_name)
        img = imageio.imread(from_images_path + img_name)
        bbs_name = img_name.replace('.jpg', '.txt')
        bbsYolo5 = pd.read_csv(from_bbs_path + bbs_name, sep=" ", header=None)
        bbsImgAug = bbsYolo5ToBbsImgAug(img, bbsYolo5)

        imgAndBb = imgAugResize(img, bbsImgAug, longerSide, False)
        imsave(to_images_path + img_name, imgAndBb[0])
        #print(from_bbs_path + bbs_name, to_bbs_path + bbs_name)
        # shutil.copy2(from_bbs_path + bbs_name, to_bbs_path +bbs_name)
        # imshow(bbsOnImage(imgAndBb[0],imgAndBb[1]))

def imgAugCropTo9Batch(images_super_path, labels_super_path, images_chopped_path, labels_chopped_path):
    file_list = os.listdir(images_super_path)
    for img_name in file_list:
        img = imageio.imread(images_super_path + img_name)
        imsave(images_chopped_path + img_name, img)
        file_name = img_name.replace('.jpg', '')
        bbs_name = file_name + '.txt'
        bbsYolo5 = pd.read_csv(labels_super_path + bbs_name, sep=" ", header=None)
        bbsYolo5.to_csv(labels_chopped_path + bbs_name, sep=' ', header=False, index=False)
        bbsImgAug = bbsYolo5ToBbsImgAug(img, bbsYolo5)
        imgAndBb = imgAugCropTo9(img, bbsImgAug, False, 400)
        i = 0
        bbs = imgAndBb[1]
        imgs = imgAndBb[0]

        for bbsImgAug in bbs:
            chopped_bbs_name = file_name + '_' + str(i) + '.txt'
            chopped_img_name = file_name + '_' + str(i) + '.jpg'
            bbsYolo5 = bbsImgAugToBbsYolo5(imgs[i], bbsImgAug)
            bbsYolo5.to_csv(labels_chopped_path + chopped_bbs_name, sep=' ', header=False, index=False)
            imsave(images_chopped_path + chopped_img_name, imgs[i])
            i = i + 1
        imshow(np.hstack(imgAndBb[0]))

def imgCrop(img,bs):
    imgs=[]
    for b in bs:
        # print(img.shape,b)
        x1=int(b.x1)
        x2=int(b.x2)
        y1=int(b.y1)
        y2=int(b.y2)
        imgCroped=img[ y1:y2, x1:x2, :]
        imgs.append(imgCroped)
    return imgs
def cropImageBatch(from_images_path, to_images_path, from_bbs_path,preview=False):
    '''
    Multi Image Resize with bbs.
    @param from_images_path:
    @param to_images_path:
    @param from_bbs_path:
    @param to_bbs_path:
    @param longerSide:
    @return:
    '''
    file_list = os.listdir(from_images_path)
    for img_name in file_list:
        #print(from_images_path + img_name)
        img = imageio.imread(from_images_path + img_name)
        bbs_name = img_name.replace('.jpg', '.txt')
        only_file_name= img_name.replace('.jpg', '')
        bbs = pd.read_csv(from_bbs_path + bbs_name, sep=" ", header=None)
        bs = bbsYolo5ToBbsImgAug(img, bbs)

        imgs = imgCrop(img, bs)
        if preview:
            ia.imshow(ia.draw_grid(imgs,  rows=1))

        print(len(imgs))
        for i in range(len(imgs)):
            imsave(to_images_path + only_file_name+"-"+str(i)+".jpg", imgs[i])

usage = ''' 
python superResolution.py {images_path} {images_super_path} 
example: 

    python imgAug.py                                    \
         /prj/nowage/ImgAugWithSR/imgs/HardHatSample__small/images_super/   \
         /prj/nowage/ImgAugWithSR/imgs/HardHatSample__small/labels_super/   \
         /prj/nowage/ImgAugWithSR/imgs/HardHatSample__small/images_chopped/ \
         /prj/nowage/ImgAugWithSR/imgs/HardHatSample__small/labels_chopped/
'''
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(usage)
    else:
        print('images_super_path         :', sys.argv[1])
        print('labels_super_path         :', sys.argv[2])
        print('images_chopped_path       :', sys.argv[3])
        print('labels_chopped_path       :', sys.argv[4])
        images_super_path    = sys.argv[1]
        labels_super_path    = sys.argv[2]
        images_chopped_path  = sys.argv[3]
        labels_chopped_path  = sys.argv[4]

        imgAugCropTo9Batch(
                images_super_path,
                labels_super_path,
                images_chopped_path,
                labels_chopped_path
        )
