# Image_Classification
* whole_pipeline.py
      *You only need to run it once, and it can automatically implement neural network training, testing, visualization, inference, and so on. Finally get the required model

* train.py
>train and return .pb model

* test.py
>You can test accuracy trained .pb model and .tflite model. It can be run in while_pipeline.py automatically and you can also do some inference using other data on already trained model.
Normally, the accuracy of .pb model will be higher, and the accuracy of quantized .tflite model will be slightly lower because of int8 format.
If you want to do inference on already trained model. Directly modify code line 104-122 

* post_training_quantization.py
>convert .pb model to quantized .tflite model.

* visualize_training_images.py
>Visualize training image. There are 4 categories in total. Each category occupies one line.

* network_model.py
>There are two functions, which are used to build model. One is MobileNet2 and another is model_builder
MobileNet2: Here we don't use hyper tunner. To freeze all the weights in the base model, only the top layer of nn can be trained and changed. Not recommended
model_builder: 3 hyperparameters will be chosen automatically. [hp_alpha, hp_training_rate, hp_layers_to_freeze]

* visualize_nn.py
>visualize nn and save in path /home/adolf/AS-21-NeuralNet/NN/models/trained/2021-07-06-17-15

* utils.py
>Generate logger and print the training situation on the terminal, and print the training process in the .txt file

* parameters.py
>Put most of parameters in this file. After modifying the parameters, do not forget to run it and write the results to hyperparameters.json
hyperparameters.json Written in from paramter.py


The following .py files are used to prepare the dataset before running whole_pipeline.py

* jpg_png_converter.py
>Your data may come from different sources. Before training and testing, we must convert them to unified format. In our data set we use .png format

* copy_remove_images.py
>Help you to copy and remove some images from one folder to another. You could do some adjustments here according to your requirement.

* image_verteilung.py
>If you have prepared a new dataset with 4 classes. blue、yellow、nocone and orange, but they are not divided into train/test/val datasets. This file can help you to divide them into train/test/val datasets according to your setup.
