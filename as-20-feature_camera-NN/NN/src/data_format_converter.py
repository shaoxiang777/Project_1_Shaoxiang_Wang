import glob
import os
import json
import numpy as np
import json
import xml.etree.ElementTree as ET
import cv2

""" 
Here there are two functions which can be used to convert different labeling file format to darknet-label.txt

If you use 'labelImg' to label images, generated annotation file will be .xml format. Please give corresponding path like this.
path_to_xml_annotations = "/home/shaoxiang/Desktop/labelImg_label_task/ann"
path_to_txt_annotations = "/home/shaoxiang/Desktop/labelImg_label_task/yolo_annotations"

If you use 'Supervisely' to label image, generated annotation file will be .json format. Please give corresponding path like this.
json_dir_path = "/home/shaoxiang/Desktop/supervisely_label Task/RaceIng Image/ann"
txt_dir_path = "/home/shaoxiang/Desktop/supervisely_label Task/RaceIng Image/txt" 

"""

# Convert xml format to yolov3 label txt format
path_to_xml_annotations = "/home/shaoxiang/Desktop/labelImg_label_task/ann"
path_to_txt_annotations = "/home/shaoxiang/Desktop/labelImg_label_task/yolo_annotations"
def xml_to_txt(path_to_xml_annotations, path_to_txt_annotations):
    if not os.path.exists(path_to_txt_annotations):
        os.makedirs(path_to_txt_annotations)

    for fp in os.listdir(path_to_xml_annotations):
        root = ET.parse(os.path.join(path_to_xml_annotations,fp)).getroot()
        size = root.find('size')
        width = float(size[0].text)
        height = float(size[1].text)

        filename = root.find('filename').text
        for child in root.findall('object'):         # Find out all bounding boxes in images
            sub = child.find('bndbox')
            xmin = float(sub[0].text)
            ymin = float(sub[1].text)
            xmax = float(sub[2].text)
            ymax = float(sub[3].text)
            label = child.find('name').text
            try:                                     # Converted to yolov3 label format, need to be normalized to the range of (0-1)
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
            except ZeroDivisionError:
                print(filename, 'has a width Error')

            with open(os.path.join(path_to_txt_annotations, fp.split('.')[0] + '.txt'), 'a+') as f:
                f.write(' '.join([label, str(x_center), str(y_center), str(w), str(h) + '\n']))


# Convert FSOCO json format to yolo label txt format
json_dir_path = "/home/adolf/Desktop/img/ann"
txt_dir_path = "/home/adolf/Desktop/img/yolo_annotations"
def json_to_txt(json_dir_path, txt_dir_path):
    if not os.path.exists(txt_dir_path):
        os.makedirs(txt_dir_path)
    json_files = sorted(
        glob.glob(os.path.join(json_dir_path, "*.json"))
    )  # get all json files in the json folder path

    for json_file in json_files:
        json_file_name = json_file.split("/")[-1]
        json_file_name = json_file_name.split(".")[0]
        json_file_name = os.path.splitext(json_file_name)[0]

        with open(json_file) as f:
            data = json.load(f)
            width = float(data["size"]["width"])
            height = float(data["size"]["height"])
            bounding_boxes = data["objects"]

            for bounding_box in bounding_boxes:
                bounding_box_class = bounding_box["classTitle"]
                bounding_box_point = bounding_box["points"]
                bounding_box_point_exterior = bounding_box_point["exterior"]
                xmin = float(bounding_box_point_exterior[0][0])
                xmax = float(bounding_box_point_exterior[1][0])
                ymin = float(bounding_box_point_exterior[0][1])
                ymax = float(bounding_box_point_exterior[1][1])

                if bounding_box_class == "blue_cone":
                    label = "0"
                elif bounding_box_class == "yellow_cone":
                    label = "1"
                elif bounding_box_class == "orange_cone":
                    label = "2"
                elif bounding_box_class == "large_orange_cone":
                    label = "3"

                try:  # Converted to yolov3 label format, need to be normalized to the range of (0-1)
                    x_center = (xmin + xmax) / (2 * width)
                    y_center = (ymin + ymax) / (2 * height)
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height
                except ZeroDivisionError:
                    print(json_file_name, "has a width Error")

                with open(
                    os.path.join(txt_dir_path, json_file_name + ".txt"), "a+"
                ) as f:
                    f.write(
                        " ".join(
                            [label, str(x_center), str(y_center), str(w), str(h) + "\n"]
                        )
                    )

# Convert darknet_json format to yolov3 label txt format
json_file_path = '/home/shaoxiang/Desktop/6.6/6.6_16_37/6.6_ka_prediction_result.json'
image_parent_folder = '/home/shaoxiang/Desktop/6.6/6.6_16_37/img'
txt_dir_path = '/home/shaoxiang/Desktop/6.6/6.6_16_37/yolo_annotations'
def supervisely_json_to_txt(json_file_path, image_parent_folder, txt_dir_path):
    if not os.path.exists(txt_dir_path):
        os.makedirs(txt_dir_path)
        
    with open(json_file_path) as f:
        data = json.load(f)

        for num in range(len(data)):
            sub_data = data[num]
            file_name = sub_data['filename'].split('/')[-1]
            image_path = os.path.join(image_parent_folder, file_name)
            file_name = file_name.split('.')[0]
            file_name = os.path.splitext(file_name)[0]

            img = cv2.imread(image_path)
            width = img.shape[1]
            height = img.shape[0]

            bounding_boxes = sub_data['objects']
            for bounding_box in bounding_boxes:
                bounding_box_class = bounding_box['name']
                relative_coordinates = bounding_box['relative_coordinates']

                x_center = relative_coordinates['center_x']
                y_center = relative_coordinates['center_y']
                w = relative_coordinates['width']
                h = relative_coordinates['height']

                # if bounding_box_class == 'blue':
                #     label = '0'
                # elif bounding_box_class == 'yellow':
                #     label = '1'
                # elif bounding_box_class == 'orange':
                #     label = '2'
                # elif bounding_box_class == 'large_orange':
                #     label = '3'

                with open(os.path.join(txt_dir_path, file_name + '.txt'), 'a+') as f:
                    f.write(' '.join([bounding_box_class, str(x_center), str(y_center), str(w), str(h) + '\n']))



def deal_with_fsoco():
    main_path = '/home/shaoxiang/Desktop/AS21/Final_Dataset_20d/fsoco_bounding_boxes_train'
    data_lists = os.listdir(main_path)
    for data_path in data_lists:
        team_path = os.path.join(main_path, data_path)
        json_dir_path =  os.path.join(team_path, 'ann')
        txt_dir_path = os.path.join(team_path, 'yolo_annotations')
        json_to_txt(json_dir_path, txt_dir_path)