import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.test.is_gpu_available()

def visualize(model, visualization_name):
    tf.keras.utils.plot_model(
        model,
        to_file=visualization_name,
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=700
    )
    print("model structure image saved to:{} ".format(visualization_name))

def generate_model_image(model_path):
    # model_path = "/home/shaoxiang/Desktop/AS21/AS-21-NeuralNet/models/2021-02-03-08-14"
    model = tf.keras.models.load_model(
        model_path
    )
    visualize(model, os.path.join(model_path, "model_visualization.png"))