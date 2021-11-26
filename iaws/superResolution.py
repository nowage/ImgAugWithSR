# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_hub as hub
from matplotlib.pyplot import imsave, imshow
import os
import sys
# import warnings
# warnings.filterwarnings("ignore")


def superResolutionDo(img_path):
    lr = tf.io.read_file(img_path)
    lr = tf.image.decode_jpeg(lr)
    lr = tf.expand_dims(lr, axis=0)
    lr = tf.cast(lr, tf.float32)

    model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")



    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape(lr.shape.as_list())
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TF Lite model.
    with tf.io.gfile.GFile('ESRGAN.tflite', 'wb') as f:
      f.write(tflite_model)

    esrgan_model_path = './ESRGAN.tflite'


    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=esrgan_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run the model
    interpreter.set_tensor(input_details[0]['index'], lr)
    interpreter.invoke()

    # Extract the output and postprocess it
    output_data = interpreter.get_tensor(output_details[0]['index'])
    sr = tf.squeeze(output_data, axis=0)
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint8)
    return sr.numpy()

# # test_img_path="/prj/nowage/ImgAugWithSR/imgs/HardHatSample__small/images/000101_jpg.rf.53b8dc83db0defce0e4fb5cd63d7ee39.jpg"
# test_img_path="/prj/nowage/ImgAugWithSR/a4.jpg"
# img=superResolutionDo(test_img_path)
# imsave('/prj/nowage/ImgAugWithSR/a4x.jpg',img)

def superResolutionBatch(images_path, images_super_path):
    file_list = os.listdir(images_path)
    for f in file_list:
        img=superResolutionDo(images_path+f)
        print(images_path+f)
        imsave(images_super_path+f,img)
        
# superResolutionBatch( images_resized_path, images_super_path )  
# !ls {images_super_path}
usage=''' 
python superResolution.py {images_path} {images_super_path} 
example: 
     
    python superResolution.py                                    \
      /prj/nowage/ImgAugWithSR/imgs/HardHatSample__small/images/ \
      /prj/nowage/ImgAugWithSR/imgs/HardHatSample__small/images_super/
'''
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(usage)
    else:
        print('images_path       :', sys.argv[1]    )
        print('images_super_path :', sys.argv[2]    )
        images_path       = sys.argv[1]
        images_super_path = sys.argv[2]
        superResolutionBatch( images_path, images_super_path )
