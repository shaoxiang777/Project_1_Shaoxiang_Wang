import os
from shutil import copy
import shutil
import random
from PIL import Image
import glob

"""
Some of the functions below are some of the functions that may be used frequently
Mainly used for copying, cutting pictures, random selection, etc.
You need to set the parameter path and function correctly according to your purpose
"""

data_base_dir_path = '/home/shaoxiang/Desktop/5'
tar_dir_path = "/home/shaoxiang/Desktop/image/New"
def rename():
    base_path = "/home/shaoxiang/Desktop/Nocone/nocone"
    tar_path = "/home/shaoxiang/Desktop/Nocone/nocone"
    filelist = os.listdir(base_path) 
    count = 0
    for file in filelist:  
        filename=os.path.splitext(file)[0]   
        filetype=os.path.splitext(file)[1]   
        old_image_path = os.path.join(base_path,file) 
        new_image_path = os.path.join(tar_path, 'nocone_nocorner_' + str(count).zfill(5) + '.jpg')  
        os.rename(old_image_path,new_image_path) 
        count+=1

def random_remove_choose():
    original_path = "/home/shaoxiang/Desktop/test 5.9/bastler 2/right/blue"
    target_path = "/home/shaoxiang/Desktop/test_bastler/blue"
    Images = os.listdir(original_path)
    samples = random.sample(Images, 1844)
    for image in samples:
        original_img_path = os.path.join(original_path,image)
        chosen_img_path =  os.path.join(target_path,image)
        shutil.move(original_img_path, chosen_img_path)

def random_copy_choose():
    original_path = "/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/old dataset/Training/yellow"
    target_path = "/home/shaoxiang/Desktop/image_verteilung/5.12 dataset/yellow"
    Images = os.listdir(original_path)
    samples = random.sample(Images, 10000)
    for image in samples:
        original_img_path = os.path.join(original_path,image)
        chosen_img_path =  os.path.join(target_path,image)
        copy(original_img_path, chosen_img_path)

# python Image length and width selector
path = '/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/old dataset/Training/orange'
def choose_image_size(path):
    # dirs = os.listdir(path)
    # for file in dirs:
    #     temp = path + "/" + file
    #     dir_path = "/home/shaoxiang/Desktop/test/test_dataset/*/*/*/*.jpg"
    dir_path = "/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/old dataset/Training/orange/*.png"
    image_path = glob.glob(dir_path)
    
    for img in image_path:
        image = Image.open(img)
        if(image.size[0] < 17 and image.size[1] < 25):
            os.remove(img)
                  
def copy_image():
    img_path = "/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/old dataset/Training/orange"
    target_path = "/home/shaoxiang/Desktop/image_verteilung/5.12 dataset/orange"
    images = os.listdir(img_path)
    for image in images:
        original_img_path = os.path.join(img_path, image)
        target_img_path = os.path.join(target_path, image)
        copy(original_img_path, target_img_path)


# Generate image path file for YOLOv4 to do object detection
def image_path_file():
    image_dir_path = '/home/shaoxiang/Desktop/6.6/21-06-06-16-37'
    images_path = sorted(glob.glob(os.path.join(image_dir_path, '*.png'))) 
    for image_path in images_path:
        image_name = image_path.split('/')[-1]
        relative_image_path = os.path.join('data/6.6_ka_img',image_name)
        with open('/home/shaoxiang/Desktop/6.6/21-06-06-16-37/6.6_ka_test.txt', 'a+') as f:
            f.write(relative_image_path)
            f.write('\n')

