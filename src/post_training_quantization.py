import os
import glob
import time
import IPython
import logging
import sys
import numpy as np
import tensorflow as tf
import network_model as net
import kerastuner as kt
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.test.is_gpu_available()

TRAIN_DIR = ""

"""
TensorFlow Lite interpreter (interpreter) is a library (library), it receives a model file (model file),
Execute the operations defined on the input data of the model file and provide access to the output.
So the following content is used to prepare input and output
"""

def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files(TRAIN_DIR + "/*/*")  
    for i in range(100):
        image = next(iter(dataset_list))    
        image = tf.io.read_file(image)
        image = tf.io.decode_image(image, channels=3)
        image = tf.image.resize(image, [net.IMG_SIZE, net.IMG_SIZE])
        image = tf.cast(image / 255.0, tf.float32)   
        image = tf.expand_dims(image, 0)   
        yield [image]                    

def quantize(tflite_model_name, model, training_dir):
    # Perform post-training integer quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)                
    # Enable qunatization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]        # The model is now a bit smaller with quantized weights, but other variable data is still in float format.
    converter.target_spec.supported_types = [tf.int8]
    # Ensure converter throws error if any operations can't be quantized
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set input and output tensors to unit8, devices that perform only integer-based operations, such as the Edge TPU.
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    # Set representative dataset to quantize activations
    global TRAIN_DIR
    TRAIN_DIR = training_dir
    converter.representative_dataset = representative_data_gen      # To quantize the variable data (such as model input/output and intermediates between layers), you need to provide representative_data_gen. Now all weights and variable data are quantized, and the model is significantly smaller compared to the original TensorFlow Lite model.
    quant_model = converter.convert()

    with tf.io.gfile.GFile(tflite_model_name, 'wb') as f:           # "tflite_model_name":"/home/shaoxiang/Desktop/AS21/AS-21-NeuralNet/models/inference/time/model_transfer_learning_bayesian_optimization.tflite",
        f.write(quant_model)
        print("Tflite model is saved to:{}".format(tflite_model_name))

    return quant_model
