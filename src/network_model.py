import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.test.is_gpu_available()

NUM_CLASSES = 4
IMG_SIZE = 64
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3) 
DROPOUT_RATE = 0.2

# This model is used if we don't use hp_tunner to find best model.
def MobileNetV2(
    input_shape,
    alpha,
    weights,
    pooling,
    dropout_rate,
    num_classes,
    learning_rate,
    loss,
    metrics
):
    feature_extractor = tf.keras.applications.MobileNetV2(
        input_shape = input_shape,
        alpha = alpha,
        weights = weights,
        pooling = pooling,
        # dropout_rate=dropout_rate,
        include_top=False,
    )
    num_layers_feat_ext = len(feature_extractor.layers)
    print("Number of layers in the feature extractor of original MobileNetV2: {}".format(num_layers_feat_ext))
    feature_extractor.trainable = False  #  to freeze all the weights in the base model.
    
    model = tf.keras.models.Sequential(
        [feature_extractor,
         tf.keras.layers.Dense(num_classes, activation = "softmax")]
    )

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
        loss = loss,
        metrics = metrics
    )

    return model

# This model is used to find best combination of hyperparameters.
def model_builder(hp):
    # Tune the learning rate for the optimizer. Choose an optimal value from 0.01, 0.001, 0.0001，or 0.00001
    hp_training_rate = hp.Choice("learning_rate", values = [1e-2, 1e-3, 1e-4, 1e-5])

    # Tune the alpha for feature_extractor. Choose an optimal value from 0.35, 0.50, 0.75, 1.0, 1.3, 1.4
    # This value has a big influence in size of model, this in turn will affect the final inference speed of the model
    hp_alpha = hp.Choice("alpha", values=[0.35, 0.50, 0.75, 1.0, 1.3, 1.4])

    feature_extractor = tf.keras.applications.MobileNetV2(
        input_shape = IMG_SHAPE,
        alpha = hp_alpha,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    num_layers_feat_ext = len(feature_extractor.layers)
    hp_layers_to_freeze = hp.Int(
        "frozen_layers", min_value = 0, max_value = num_layers_feat_ext, default = 100    
    )
    for layer in feature_extractor.layers[:hp_layers_to_freeze]: 
        layer.trainable = False
    model = tf.keras.models.Sequential(
        [feature_extractor, 
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = hp_training_rate),
        loss = 'categorical_crossentropy',        
        metrics = ["accuracy"],
    )

    return model



