from PIL import Image
import cv2 as cv
import os

# convert .png to .jpg
def png_jpg(png_path):
    img = cv.imread(png_path, 0)
    w, h = img.shape[::-1]
    infile = png_path

    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    img = img.resize((int(w), int(h)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=100)
            os.remove(png_path)
        else:
            img.convert('RGB').save(outfile, quality=100)
            os.remove(png_path)
        return outfile
    except Exception as e:
        print("PNG convert JPG error", e)

# convert .jpg to .png
def jpg_png(jpg_path):
    img = cv.imread(jpg_path, 0)
    w, h = img.shape[::-1]
    infile = jpg_path

    outfile = os.path.splitext(infile)[0] + ".png"
    img = Image.open(infile)
    img = img.resize((int(w), int(h)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=100)
            os.remove(jpg_path)
        else:
            img.convert('RGB').save(outfile, quality=100)
            os.remove(jpg_path)
        return outfile
    except Exception as e:
        print("JPG convert PNG error", e)

def main():
    jpg_to_png = False
    path_root = os.getcwd()
    path ='/home/shaoxiang/Desktop/image_converter/'
    img_dir = os.listdir(path)
    if jpg_to_png == True:
        for img in img_dir:
            if img.endswith('.jpg'):
                jpg_path= path + img
                jpg_png(jpg_path)
        img_dir = os.listdir(path)
    else:
        for img in img_dir:
            if img.endswith('.png'):
                png_path= path  + img
                png_jpg(png_path)
        img_dir = os.listdir(path)
    for img in img_dir:
        print(img)

if __name__ == "__main__":
    main()