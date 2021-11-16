import os
import glob
import time
import IPython
import logging
import sys
import numpy as np
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.test.is_gpu_available()
import network_model as net

import kerastuner as kt

def set_input_tensor(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    # Inputs for the TFLite model must be uint8, so we quantize our input data.
    # NOTE: This step is necessary only because we're receiving input data from
    # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
    # bitmap inputs, they're already uint8 [0,255] (we actually use int8) so this can be replaced with:
    #   input_tensor[:, :] = input
    scale, zero_point = input_details["quantization"]  # scale is 0.00392 = 1/255  zero_point is -128
    input_tensor[:, :] = np.uint8(input / scale + zero_point)  # Now the data here is the type of uint8

def classify_image(interpreter, input):
    set_input_tensor(interpreter, input)  # 之前的input是batch_images[i]，他还不是一个unit8的形式，之后就变成了uint8 的类型了
    interpreter.invoke()                  # 这句话可以理解为让interpreter开始工作，处理后输出output
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])
    # outputs from the TFLite model are uint8, so results need to be dequantized
    scale, zero_point = output_details["quantization"]
    output = (output - zero_point) * scale
    top_1 = np.argmax(output)
    return top_1

def test(
    model, img_size, test_dir, model_path=None, quantized = False
):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0 / 255)
    test_generator = test_datagen.flow_from_directory(    
        test_dir,
        batch_size=32 ,
        target_size=(img_size, img_size),
        class_mode="categorical",         
        interpolation="bilinear",
    )
    if model_path is not None:
        model = tf.keras.models.load_model(model_path)  
    if not quantized:
        loss_value, model_test_accuracy = model.evaluate(
            test_generator, steps = len(test_generator)  
        )
        print("Baseline test accuracy: {:.3%}".format(model_test_accuracy))
       
    else:
        TFLite_accuracy = []
        for i in range(len(test_generator)):
            batch_images, batch_labels  = next(iter(test_generator))        # Here batch_images and batch_labels are both a list
            #  Initialize the interpreter
            interpreter = tf.lite.Interpreter(model)
            # interpreter = tflite.Interpreter(model, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            interpreter.allocate_tensors()

            # Collect all inference predictions in a list
            batch_prediction = []
            batch_truth = np.argmax(batch_labels, axis=1)

            for i in range(len(batch_images)):
                prediction = classify_image(interpreter, batch_images[i])   # Every time you enter classify_image is a image
                batch_prediction.append(prediction)

            # Compare all predictions to the ground truth
            tflite_accuracy = tf.keras.metrics.Accuracy()
            tflite_accuracy(batch_prediction, batch_truth)
            TFLite_accuracy.append(tflite_accuracy.result().numpy())
        mean_TFLite_accuracy = np.mean(TFLite_accuracy)
        
        # print("Quant TF Lite accuracy: {:.3%}".format(mean_TFLite_accuracy))
        print("TFlite model accuracy on test data set : {:.3%}".format(mean_TFLite_accuracy))




