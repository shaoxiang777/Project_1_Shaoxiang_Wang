import os
import cv2
import numpy as np
import imghdr
import random

path_to_label_task = "/home/shaoxiang/Desktop/new"
path_to_images = os.path.join(path_to_label_task, "img")
path_to_annotations = os.path.join(path_to_label_task, "yolo_annotations")
path_to_cropped_out_images = os.path.join(path_to_label_task, "image_classification")
path_to_classes = os.path.join(path_to_label_task, "classes.txt")
classes = []

# a positive number indicates an offset in that direction
upper_random_list = [0.5]
lower_random_list = [0.5]
left_random_list = [0.5]
right_random_list = [0.5]

def crop_image():
    # Create destination folder if not already present
    if not os.path.exists(path_to_cropped_out_images):
        os.mkdir(path_to_cropped_out_images)
    # Determine labels (classes)
    with open(path_to_classes, 'r') as classes_file:
        for line in classes_file:
            class_label = line.strip()
            classes.append(class_label)
            # Create folder for each class
            if not os.path.exists(os.path.join(path_to_cropped_out_images, class_label)):
                os.mkdir(os.path.join(path_to_cropped_out_images, class_label))
    # Iterate through annotated images
    for original_img_object in os.scandir(path_to_images):
        # Ignore directory
        if not os.path.isdir(original_img_object):
            original_img = original_img_object.name
            original_img_name = os.path.splitext(original_img)[0] # Separate file name and extension; return (fname, fextension) tuple 
            # Ignore files which are not images
            if imghdr.what(os.path.join(path_to_images, original_img)) is not None:
                img = cv2.imread(os.path.join(path_to_images, original_img))
                image_height, image_width = img.shape[:2]
                # If corresponding .txt file containing annotations exists, go through it
                corresponding_annotation_file = os.path.join(
                    path_to_annotations, "{}.txt".format(original_img_name))
                print(corresponding_annotation_file)
               
                if os.path.isfile(corresponding_annotation_file):
                    with open(corresponding_annotation_file, "r") as bounding_boxes_file:
                        counter = 0
                        # Iterate through bounding_boxes
                        for bounding_box in bounding_boxes_file:
                            # Extract information from YOLO label: <object-class> <x_center> <y_center> <width> <height>
                            annotation = list(
                                 bounding_box.split())
                            class_label = annotation[0]
                            x_center = float(annotation[1])
                            absolute_x = image_width * x_center
                            y_center = float(annotation[2])
                            absolute_y = image_height * y_center
                            bounding_box_width = float(annotation[3])
                            absolute_width = image_width * bounding_box_width
                            bounding_box_height = float(annotation[4])
                            absolute_height = image_height * bounding_box_height
                            
                            # Crop out image
                            lower_boundary = int(absolute_y - absolute_height / 2.0) 
                            upper_boundary = int(absolute_y + absolute_height / 2.0)
                            left_boundary = int(absolute_x - absolute_width / 2.0) 
                            right_boundary = int(absolute_x + absolute_width / 2.0)

                            # lower_random_list = [0]
                            # upper_random_list = [0]
                            # left_random_list = [0]
                            # right_random_list = [0]

                            
                            lower_trans = random.choice(lower_random_list)
                            upper_trans = random.choice(upper_random_list)
                            left_trans = random.choice(left_random_list)
                            right_trans = random.choice(right_random_list)

                            lower_boundary = lower_boundary - int(lower_trans * absolute_height)
                            upper_boundary = upper_boundary + int(upper_trans * absolute_height)
                            left_boundary = left_boundary - int(left_trans * absolute_width)
                            right_boundary = right_boundary + int(right_trans * absolute_width) 
                             
                            # Prevent beyond the boundaries of the image
                            for boundary in [lower_boundary, upper_boundary, left_boundary, right_boundary]:
                                while lower_boundary < 0:
                                    lower_boundary += 1
                                while upper_boundary > image_height:
                                    upper_boundary += -1
                                while left_boundary < 0:
                                    left_boundary += 1
                                while right_boundary > image_width:
                                    right_boundary += -1
                                
                            crop_img_bounding_box = img[lower_boundary : upper_boundary,
                                                       left_boundary : right_boundary].copy()
                            cv2.imshow("cropped image", crop_img_bounding_box)
                            # Save picture to image_classification/class/
                            cv2.imwrite(os.path.join(path_to_cropped_out_images, class_label, "{}-{}.png".format(original_img_name,counter)),
                                        crop_img_bounding_box)
                            print(os.path.join(path_to_cropped_out_images, class_label, "{}-{}.png".format(original_img_name,counter)))
                            print("{} cropped image saved".format(counter))
                            counter += 1


# =========================
# # This is used to crop out fsoco all images

# # main_path = '/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/fsoco_bounding_boxes_train'
# # data_lists = os.listdir(main_path)
# # for data_path in data_lists:
# #     path_to_label_task = os.path.join(main_path, data_path)
# #     path_to_images = os.path.join(path_to_label_task, "img")
# #     path_to_annotations = os.path.join(path_to_label_task, "yolo_annotations")
# #     path_to_cropped_out_images = os.path.join(path_to_label_task, "image_classification")
# #     path_to_classes = os.path.join(path_to_label_task, "classes.txt")
# #     classes = []

# #     crop_image()

# =========================
