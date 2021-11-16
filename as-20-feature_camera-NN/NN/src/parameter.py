import json

data = {
    # parameters for MobileNetV2
    "image_size":64,
    "alpha":1.0,
    "weight":"imagenet",
    "pooling":"avg",
    "dropout_rate":0.2,
    "learning_rate":1e-3,
    "num_classes":4,
    "optimizer":"adam",
    "loss":"categorical_crossentropy",
    "metrics":"accuracy",

    "activation":"relu",
    "batch_size":32,
    "epochs":30,

    # parameters for early_stop, reduce_lr_plateau
    "monitor":"val_loss",
    "reduce_lr_patience":7,
    "early_stop_patience":10,
    
    # parameters for ImageDataGenerator (data augmentation)
    "data_aug_rotation":0,
    "data_aug_width_shift":0.2,
    "data_aug_height_shift":0.2,
    "data_aug_min_brightness":0.4,
    "data_aug_max_brightness":1.0,
    "data_aug_zoom":0.2,
    "data_aug_horizontal_flip":True,
    "data_aug_fill":"constant",
    "gen_class_mode":"categorical",
    "gen_interpolation":"bilinear",

    "hp_tuning":True,
    "hp_tuner":"bayesian_optimization",
    "hp_dir":"/home/shaoxiang/Desktop/NN/logs/bayesian_optimization", # Some pre-trained trials are stored here
    "hp_project_name":"transfer_learning_bayesian_optimization", # Or custom_network_hp_tuning_bayesian_optimization
    "log_dir":"/home/shaoxiang/Desktop/NN/logs/fit",   # Here it will store 'train' and 'validation' for every train
    "logger_name":"train.log",

    # divide datasets
    "training_dir":"/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/new dataset/train",
    "val_dir":"/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/new dataset/val",
    "test_dir":"/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/new dataset/test",

    # save trained .pb model and .tflite model
    "train_imgs_vis_name":"visualize_training_images.png",
    "model_dir":"/home/shaoxiang/Desktop/NN/models/trained", # Here is trained model.pb
    "tflite_model_path":"/home/shaoxiang/Desktop/NN/models/inference", 
    "tflite_model_name":"model_transfer_learning_bayesian_optimization.tflite",
    "edgetpu_output_dir":"/home/shaoxiang/Desktop/NN/models/inference", # Here is inferenced model.tflite
}

json_path = "/home/shaoxiang/Desktop/NN/src/hyperparameters.json"
# Writing JSON data
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)