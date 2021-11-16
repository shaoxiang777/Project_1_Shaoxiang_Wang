import os
from shutil import copy
import shutil
import random
from PIL import Image
import glob

# data_base_dir_path = '/home/shaoxiang/Desktop/5'
# tar_dir_path = "/home/shaoxiang/Desktop/image/New"


def rename():
    base_path = "/home/shaoxiang/Desktop/6.6/6.6_16_37/image_classification/yellow_cone"
    tar_path = "/home/shaoxiang/Desktop/6.6/6.6_16_37/image_classification/yellow_cone"
    filelist = os.listdir(base_path) 
    # filelist.sort(key=lambda x:int(x[-9:-4]))
    count = 6238
    for file in filelist:  
        filename=os.path.splitext(file)[0]   
        filetype=os.path.splitext(file)[1]   
        old_image_path = os.path.join(base_path,file) 
        new_image_path = os.path.join(tar_path, 'rainy_nocone_' + str(count).zfill(5) + '.png')  
        os.rename(old_image_path,new_image_path) 
        count+=1


def random_remove_choose():
    original_path = "/home/shaoxiang/Desktop/fosoco_all_divide_image/large_orange_cone"
    target_path = "/home/shaoxiang/Desktop/fosoco_all_divide_image/orange_cone"
    Images = os.listdir(original_path)
    samples = random.sample(Images, 4238)
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

def copy_fsoco_data():
    img_path = '/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/fsoco_bounding_boxes_train'
    blue_target_path = '/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/fsoco_bounding_boxes_train/ampera/target/blue_cone'
    yellow_target_path = '/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/fsoco_bounding_boxes_train/ampera/target/yellow_cone'
    large_orange_target_path = '/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/fsoco_bounding_boxes_train/ampera/target/large_orange_cone'
    orange_target_path = '/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/fsoco_bounding_boxes_train/ampera/target/orange_cone'

    uni_folders = os.listdir(img_path)
    for uni_folder in uni_folders:
        uni_folder_path = os.path.join(img_path, uni_folder)
        uni_image_path = os.path.join(uni_folder_path, 'image_classification')
        uni_blue_image_path = os.path.join(uni_image_path, 'blue_cone')
        uni_yellow_image_path = os.path.join(uni_image_path, 'yellow_cone')
        uni_large_orange_image_path = os.path.join(uni_image_path, 'large_orange_cone')
        uni_orange_image_path = os.path.join(uni_image_path, 'orange_cone')

        blue_images = os.listdir(uni_blue_image_path)
        for image in blue_images:
            original_img_path = os.path.join(uni_blue_image_path, image)
            target_img_path = os.path.join(blue_target_path, image)
            copy(original_img_path, target_img_path)
        
        yellow_images = os.listdir(uni_yellow_image_path)
        for image in yellow_images:
            original_img_path = os.path.join(uni_yellow_image_path, image)
            target_img_path = os.path.join(yellow_target_path, image)
            copy(original_img_path, target_img_path)

        large_orange_images = os.listdir(uni_large_orange_image_path)
        for image in large_orange_images:
            original_img_path = os.path.join(uni_large_orange_image_path, image)
            target_img_path = os.path.join(large_orange_target_path, image)
            copy(original_img_path, target_img_path)

        orange_images = os.listdir(uni_orange_image_path)
        for image in orange_images:
            original_img_path = os.path.join(uni_orange_image_path, image)
            target_img_path = os.path.join(orange_target_path, image)
            copy(original_img_path, target_img_path)























