# -*- coding: utf-8 -*

'''
If you have prepared a new dataset with 4 classes. blue、yellow、nocone and orange.
This file can help you to divide them into train/test/val datasets according to your setup.
'''

import os
import random
import shutil

original_dataset_folder_path = '/home/shaoxiang/Desktop/fosoco_all_divide_image'
original_blue_path = os.path.join(original_dataset_folder_path, 'blue')
original_yellow_path = os.path.join(original_dataset_folder_path, 'yellow')
original_nocone_path = os.path.join(original_dataset_folder_path, 'nocone')
original_orange_path = os.path.join(original_dataset_folder_path, 'orange')

# Give the proportion of division
train_rate = 0.7
val_rate = 0.2 
test_rate = 0.1
total_blue_number = len(os.listdir(original_blue_path))
total_yellow_number = len(os.listdir(original_yellow_path))
total_nocone_number = len(os.listdir(original_nocone_path))
total_orange_number = len(os.listdir(original_orange_path))

train_blue_number = int(total_blue_number * train_rate)
train_yellow_number = int(total_yellow_number * train_rate)
train_nocone_number = int(total_nocone_number * train_rate)
train_orange_number = int(total_orange_number * train_rate)

val_blue_number = int(total_blue_number * val_rate)
val_yellow_number = int(total_yellow_number * val_rate)
val_nocone_number = int(total_nocone_number * val_rate)
val_orange_number = int(total_orange_number * val_rate)

test_blue_number = total_blue_number - train_blue_number - val_blue_number
test_yellow_number = total_yellow_number - train_yellow_number - val_yellow_number
test_nocone_number = total_nocone_number - train_nocone_number - val_nocone_number
test_orange_number = total_orange_number - train_orange_number - val_orange_number



dataset_name = 'divided_dataset'
divided_dataset_path = os.path.join(original_dataset_folder_path, dataset_name)
os.mkdir(divided_dataset_path)

sub_datasets = ['train', 'val', 'test']
color_folders = ['blue', 'yellow', 'nocone', 'orange']

def image_verteilung():
    for sub_dataset in sub_datasets:
        sub_dataset_path = os.path.join(divided_dataset_path, sub_dataset)
        os.mkdir(sub_dataset_path)
        for color_folder in color_folders:
            color_folder_path = os.path.join(sub_dataset_path, color_folder)
            os.mkdir(color_folder_path)

            # divide blue folder
            if sub_dataset =='train' and color_folder == 'blue':
                Images = os.listdir(original_blue_path)
                samples = random.sample(Images, train_blue_number)
                for image in samples:
                    original_img_path = os.path.join(original_blue_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)
            elif sub_dataset == 'val' and color_folder == 'blue':
                Images = os.listdir(original_blue_path)
                samples = random.sample(Images, val_blue_number)
                for image in samples:
                    original_img_path = os.path.join(original_blue_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)
            elif sub_dataset == 'test' and color_folder == 'blue':
                Images = os.listdir(original_blue_path)
                samples = random.sample(Images, test_blue_number)
                for image in samples:
                    original_img_path = os.path.join(original_blue_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)

            # divide yellow folder
            if sub_dataset =='train' and color_folder == 'yellow':
                Images = os.listdir(original_yellow_path)
                samples = random.sample(Images, train_yellow_number)
                for image in samples:
                    original_img_path = os.path.join(original_yellow_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)
            elif sub_dataset == 'val' and color_folder == 'yellow':
                Images = os.listdir(original_yellow_path)
                samples = random.sample(Images, val_yellow_number)
                for image in samples:
                    original_img_path = os.path.join(original_yellow_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)
            elif sub_dataset == 'test' and color_folder == 'yellow':
                Images = os.listdir(original_yellow_path)
                samples = random.sample(Images, test_yellow_number)
                for image in samples:
                    original_img_path = os.path.join(original_yellow_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)

            # divide nocone folder
            if sub_dataset =='train' and color_folder == 'nocone':
                Images = os.listdir(original_nocone_path)
                samples = random.sample(Images, train_nocone_number)
                for image in samples:
                    original_img_path = os.path.join(original_nocone_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)
            elif sub_dataset == 'val' and color_folder == 'nocone':
                Images = os.listdir(original_nocone_path)
                samples = random.sample(Images, val_nocone_number)
                for image in samples:
                    original_img_path = os.path.join(original_nocone_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)
            elif sub_dataset == 'test' and color_folder == 'nocone':
                Images = os.listdir(original_nocone_path)
                samples = random.sample(Images, test_nocone_number)
                for image in samples:
                    original_img_path = os.path.join(original_nocone_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)

            # divide orange folder
            if sub_dataset =='train' and color_folder == 'orange':
                Images = os.listdir(original_orange_path)
                samples = random.sample(Images, train_orange_number)
                for image in samples:
                    original_img_path = os.path.join(original_orange_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)
            elif sub_dataset == 'val' and color_folder == 'orange':
                Images = os.listdir(original_orange_path)
                samples = random.sample(Images, val_orange_number)
                for image in samples:
                    original_img_path = os.path.join(original_orange_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)
            elif sub_dataset == 'test' and color_folder == 'orange':
                Images = os.listdir(original_orange_path)
                samples = random.sample(Images, test_orange_number)
                for image in samples:
                    original_img_path = os.path.join(original_orange_path, image)
                    chosen_img_path = color_folder_path
                    shutil.move(original_img_path, chosen_img_path)

if __name__ == "__main__":
    image_verteilung()






