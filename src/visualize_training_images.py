import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def visualize(training_dir, visualization_name):
    blue_dir = os.path.join(training_dir, "blue")
    blue_fnames = os.listdir(blue_dir)
    chosen_blue_fnames = random.sample(blue_fnames, 8)
    nocone_dir = os.path.join(training_dir, "nocone")
    nocone_fnames = os.listdir(nocone_dir)
    chosen_nocone_fnames = random.sample(nocone_fnames, 8)
    orange_dir = os.path.join(training_dir, "orange")
    orange_fnames = os.listdir(orange_dir)
    chosen_orange_fnames = random.sample(orange_fnames, 8)
    yellow_dir = os.path.join(training_dir, "yellow")
    yellow_fnames = os.listdir(yellow_dir)
    chosen_yellow_fnames = random.sample(yellow_fnames, 8)

    # Parameters for our graph; we'll output images in a 3*8 configuration
    nrows = 4
    ncols = 8

    # Set up matplotlib fig, and size it to fit 3*8 pics
    fig = plt.gcf()  
    fig.set_size_inches(ncols*20, nrows*20)

    next_blue_pix = [
        os.path.join(blue_dir, fname)
        for fname in chosen_blue_fnames
    ]              

    next_nocone_pix = [
        os.path.join(nocone_dir, fname)
        for fname in chosen_nocone_fnames
    ]

    next_orange_pix = [
        os.path.join(orange_dir, fname)
        for fname in chosen_orange_fnames
    ]

    next_yellow_pix = [
        os.path.join(yellow_dir, fname)
        for fname in chosen_yellow_fnames
    ]

    for i, img_path in enumerate(next_blue_pix + next_orange_pix + next_yellow_pix):
        # Set up subplot； subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis("Off")  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.savefig(visualization_name, bbox_inches="tight")  # Bbox means bounding box, tight is to make the saved picture tighter


def main():
    visualize(
        training_dir = "/home/shaoxiang/Desktop/test_bastler",
        visualization_name = "visualize_training_images_5_16.png"
    )
