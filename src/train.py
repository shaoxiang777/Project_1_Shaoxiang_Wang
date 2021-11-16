import os
import glob
import time
import logging
import IPython
import sys
import json
import tensorflow as tf
import kerastuner as kt
from utils import Params
from utils import set_logger
from shutil import copyfile
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.test.is_gpu_available()


def train(
    model,
    image_size,
    hp_tuning,
    hp_tuner,
    hp_model,                                    
    batch_size,
    epochs,
    early_stop_patience,
    hp_dir,
    hp_project_name,
    training_dir,
    val_dir,
    test_dir,
    log_dir,
    logger_name,
    time
):
    json_path = "/home/shaoxiang/Desktop/NN/src/hyperparameters.json"
    HP_DIR_PROJECT = os.path.join(hp_dir, hp_project_name)

    train_blue_dir = os.path.join(training_dir, 'blue')
    train_blue_num = len(os.listdir(train_blue_dir))
    train_nocone_dir = os.path.join(training_dir, 'nocone')
    train_nocone_num = len(os.listdir(train_nocone_dir))
    train_orange_dir = os.path.join(training_dir, 'orange')
    train_orange_num = len(os.listdir(train_orange_dir))
    train_yellow_dir = os.path.join(training_dir, 'yellow')
    train_yellow_num = len(os.listdir(train_yellow_dir))
    total_train_num = train_blue_num + train_nocone_num + train_orange_num + train_yellow_num

    val_blue_dir = os.path.join(val_dir, 'blue')
    val_blue_num = len(os.listdir(val_blue_dir))
    val_nocone_dir = os.path.join(val_dir, 'nocone')
    val_nocone_num = len(os.listdir(val_nocone_dir))
    val_orange_dir = os.path.join(val_dir, 'orange')
    val_orange_num = len(os.listdir(val_orange_dir))
    val_yellow_dir = os.path.join(val_dir, 'yellow')
    val_yellow_num = len(os.listdir(val_yellow_dir))
    total_val_num = val_blue_num + val_nocone_num + val_orange_num + val_yellow_num


    test_blue_dir = os.path.join(test_dir, "blue")
    test_blue_num = len(os.listdir(test_blue_dir))
    test_nocone_dir = os.path.join(test_dir, "nocone")
    test_nocone_num = len(os.listdir(test_nocone_dir))
    test_orange_dir = os.path.join(test_dir, "orange")
    test_orange_num = len(os.listdir(test_orange_dir))
    test_yellow_dir = os.path.join(test_dir, "yellow")
    test_yellow_num = len(os.listdir(test_yellow_dir))
    total_test_num = test_blue_num + test_nocone_num + test_orange_num + test_yellow_num

    if not hp_tuning: # This situation is rarely used
        if not os.path.exists(log_dir):                       
            os.mkdir(log_dir) 
        LOGGER = os.path.join(log_dir, logger_name)
        set_logger(LOGGER)                              
        logging.info("Captain's Log\n")
        # Log hyperparameters
        copyfile(json_path, os.path.join(log_dir, "hyperparameters.json"))
    else:
        # Create destination folder if not present
        if not os.path.exists(HP_DIR_PROJECT):
            os.mkdir(HP_DIR_PROJECT)
        LOGGER = os.path.join(HP_DIR_PROJECT, logger_name)
        # If there is no such .log file in this file path, it will be generated directly. If it already exists, continue to write under it
        set_logger(LOGGER)              
        logging.info("Captain's Log\n")
        # Log hyperparameters
        copyfile(json_path, os.path.join(HP_DIR_PROJECT, "hyperparameters.json"))  
    
    # Load the additional parameters needed from json file
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    training_dir = params.training_dir
    val_dir = params.val_dir
    train_image_path = training_dir + "/*/*.png"
    val_image_path = val_dir + "/*/*.png"
    train_image_path = glob.glob(train_image_path)
    val_image_path = glob.glob(val_image_path)

    train_image_label = []
    for p in train_image_path:
        image_class = p.split('/')[-2]
        if image_class == 'blue':
            train_image_label.append([1,0,0,0])
        elif image_class == 'nocone':
            train_image_label.append([0,1,0,0])
        elif image_class == 'orange':
            train_image_label.append([0,0,1,0])
        elif image_class == 'yellow':
            train_image_label.append([0,0,0,1])
        

    val_image_label = []
    for p in val_image_path:
        image_class = p.split('/')[-2]
        if image_class == 'blue':
            val_image_label.append([1,0,0,0])
        elif image_class == 'nocone':
            val_image_label.append([0,1,0,0])
        elif image_class == 'orange':
            val_image_label.append([0,0,1,0])
        elif image_class == 'yellow':
            val_image_label.append([0,0,0,1])
        
    
    def load_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels = 3)
        image = tf.image.resize(image, [64,64])
        
    #     image = tf.image.random_crop(image,[256,256,3])
    #     image = tf.image.random_flip_left_right(image)
    #     image = tf.image.random_flip_up_down(image)
    #     image = tf.image.random_brightness(image, 0.5)
    #     image = tf.image.random_contrast(image, 0.1)
        
        image = tf.cast(image, tf.float32)
        image = image/255

        return image,label

    train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    val_image_ds = tf.data.Dataset.from_tensor_slices((val_image_path, val_image_label))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_image_ds = train_image_ds.map(load_preprocess_image,num_parallel_calls = AUTOTUNE)
    val_image_ds = val_image_ds.map(load_preprocess_image,num_parallel_calls = AUTOTUNE)

    BATCH_SIZE = batch_size
    train_acount = len(train_image_path)
    val_acount = len(val_image_ds)
    train_image_ds = train_image_ds.shuffle(train_acount).repeat().batch(BATCH_SIZE)
    train_image_ds = train_image_ds.prefetch(AUTOTUNE)

    val_image_ds = val_image_ds.batch(BATCH_SIZE)
    val_image_ds = val_image_ds.prefetch(AUTOTUNE) 

    imgs, labels = next(iter(train_image_ds))
    val_imgs, val_labels = next(iter(val_image_ds))

    steps_per_epoch = train_acount//BATCH_SIZE
    validation_steps = val_acount//BATCH_SIZE

    # Dynamic images for standard evaluation of visual testing and training   Instructions： tensorboard --logdir=/full_path_to_your_logs
    tensor_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir,
        histogram_freq = 1, 
        write_graph = True,
        write_images = True, 
        update_freq = "epoch"
    )

    # Reduce learning rate when a metric has stopped improving.
    reduce_lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=params.monitor,
        factor=0.1,                           # new_lr = lr * factor
        patience=params.reduce_lr_patience,   
        verbose=1,                           
        mode="auto",
        min_delta=0.0001,
        cooldown=0,                          
        min_lr=0
    )

    # Stop training when a metric has stopped improving.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=params.monitor,   
        patience=params.early_stop_patience,  # in general params.reduce_lr_patience < params.early_stop_patience
        verbose=1,
        mode="auto",  
        restore_best_weights=True 
    )

    class ClearTrainingOutput(tf.keras.callbacks.Callback):
        def on_train_end(*args, **kwargs):
            IPython.display.clear_output(wait=True)

    logging.info("Choose Hypertuning: {}".format(hp_tuning))
    if hp_tuning:
        clear_train_output = ClearTrainingOutput()
        if hp_tuner == "hyperband":
            logging.info("hp tuner: {}".format(hp_tuner))
            tuner = kt.Hyperband(
                hp_model,
                objective = params.monitor,
                max_epochs = epochs, 
                factor = 3, 
                directory = hp_dir,
                project_name = hp_project_name, 
            )
        else:
            logging.info("hp tuner: {}".format(hp_tuner))
            tuner = kt.BayesianOptimization(
                hp_model,                                    
                objective=params.monitor,
                max_trials = 30,                              # Total number of trials (model configurations) to test at most.
                directory = hp_dir,
                project_name = hp_project_name
            )
        tuner.search(
                train_image_ds,
                epochs = epochs,
                steps_per_epoch = steps_per_epoch,
                validation_data = val_image_ds,
                validation_steps = validation_steps,
                callbacks = [clear_train_output]             # After each end, call clear
            )
        
        # Print and log the result of Tuner
        logging.info("Tuner results：\n")
        sys.stdout = open(LOGGER, "a")                       # Redirect stdout because summary() functions do not log properly 
        tuner.results_summary() 
        best_hps = tuner.get_best_hyperparameters()[0]
        model = tuner.hypermodel.build(best_hps)             # Use the best hy to build this model, the essence!
        model.summary() 
        sys.stdout = sys.__stdout__
    else:
        model.summary()
    logging.info("Number of layers in the model: {}".format(len(model.layers)))
    logging.info("Number of trainable layers weights: {}".format(len(model.trainable_weights)))
    logging.info("Total training images: {}\n".format(total_train_num))
    logging.info("Total validation images: {}\n".format(total_val_num))
    logging.info("Total test images: {}\n".format(total_test_num))
    logging.info("Total images: {}\n".format(total_train_num + total_val_num + total_test_num))


    model.fit(
        train_image_ds, 
        epochs = epochs, 
        steps_per_epoch = steps_per_epoch,
        validation_steps = validation_steps, 
        validation_data = val_image_ds,
        verbose = 1,
        callbacks = [tensor_callback, early_stop, reduce_lr_plateau],
    )

    # save trained model
    export_trained_model_path = "{}/{}".format(params.model_dir, time) 
    logging.info("Model exported to {}".format(export_trained_model_path))
    model.save(export_trained_model_path)
    return model










