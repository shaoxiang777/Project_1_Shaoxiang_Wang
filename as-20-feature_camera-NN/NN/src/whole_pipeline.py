import os
import tensorflow as tf

import glob
import time
import logging
import json

import network_model as net
import visualize_nn as vis
from train import train
import visualize_training_images as vis_train_imgs 
from post_training_quantization import quantize
from test import test
from utils import Params  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.test.is_gpu_available()
print(tf.test.is_gpu_available())

def main():
    t = time.time()
    local_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(t))

    #Load the parameters from json file
    json_path = '/home/shaoxiang/Desktop/NN/src/hyperparameters.json'
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = Params(json_path)

    LOG_DIR = os.path.join(params.log_dir, local_time)
    saved_model_dir = os.path.join(params.model_dir, local_time)

    # Train network
    trained_model = train(
        model = net.MobileNetV2(
            input_shape = (params.image_size, params.image_size, 3),
            alpha = params.alpha,
            weights = params.weight,
            pooling = params.pooling,
            dropout_rate = params.dropout_rate, 
            num_classes = params.num_classes,
            learning_rate = params.learning_rate,
            loss = params.loss,
            metrics = params.metrics
        ),
        image_size = params.image_size,
        hp_tuning = params.hp_tuning,
        hp_tuner = params.hp_tuner,
        hp_model = net.model_builder,                                    
        batch_size = params.batch_size,
        epochs = params.epochs,
        early_stop_patience = params.early_stop_patience,
        hp_dir = params.hp_dir,
        hp_project_name = params.hp_project_name,
        training_dir = params.training_dir,
        val_dir = params.val_dir,
        test_dir = params.test_dir,
        log_dir = LOG_DIR,
        logger_name = params.logger_name,
        time = local_time
    )

    # If hyperparameters were optimized, create a CSV file to visualize hparams using Hiplot
    if params.hp_tuning:  
        os.system(
            "python3 /home/shaoxiang/Desktop/NN/src/keras_tuner_to_hip.py {}".format(
                os.path.join(params.hp_dir, params.hp_project_name) 
            )
        )

    # # Visualize training images
    vis_train_imgs.visualize(
        training_dir = params.training_dir,
        visualization_name = os.path.join(LOG_DIR, params.train_imgs_vis_name)   
    )

    # Visualize model's layers/structure
    vis.generate_model_image(saved_model_dir)

    # Fully quantize model and convert to TensorFlow Lite
    tflite_model_path = os.path.join(params.tflite_model_path, local_time)
    os.mkdir(tflite_model_path)
    tflite_model_name = os.path.join(tflite_model_path, params.tflite_model_name)

    quant_trained_model = quantize(
        tflite_model_name=tflite_model_name,
        model = trained_model,
        training_dir=params.training_dir
    )

    # Test baseline model and quantized model
    test(model = trained_model, 
         img_size = params.image_size, 
         test_dir = params.test_dir)
    test(
        model = tflite_model_name,
        img_size = params.image_size,
        test_dir = params.test_dir,
        quantized = True
    )
    print("\n"*5)

    # Compile quantized TFLite model to execute using EdgeTPU
    edgetpu_output_dir = os.path.join(params.edgetpu_output_dir, local_time)
    os.system(
        "edgetpu_compiler -s {} -o {}".format(
            tflite_model_name, edgetpu_output_dir
        )
    )

if __name__ == "__main__":
    main()













