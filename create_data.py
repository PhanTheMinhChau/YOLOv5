import glob
import os
from PIL import Image

import cv2
from random import *
import numpy as np
import string
import tqdm

#Đường dẫn đến thư mục background và characters
BG_PATH = './bg/*.jpg'
CHAR_PATH = './char/*.png'

#Lấy ra tên file ảnh
bg_items = glob.glob(BG_PATH)
char_items = glob.glob(CHAR_PATH)

#Size ảnh
SIZE = (500, 500)

#Scale của các characters
SCALE = [0.25 , 0.5, .75, 1 , 1.25]



def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """Random tên file

    Args:
        size (int, optional): [description]. Defaults to 6.
        chars ([type], optional): [description]. Defaults to string.ascii_uppercase+string.digits.

    Returns:
        [type]: [description]
    """
    return ''.join(choice(chars) for _ in range(size))

def build_data(n_images, mode):
    """Tạo data

    Args:
        n_images (int): Số sample
        mode (string): train / valid / test
    """
    #Dùng tqdm để hiển thị process bar
    t = tqdm.tqdm(range(n_images))
    for i in t:
        try:
            #Đọc ảnh và class_id
            background   = Image.open(choice(bg_items))
            img_path = choice(char_items)
            class_id = img_path.split('\\')[-1].split(".")[0]
            char = Image.open(img_path)
        except:
            continue

        #random scale cho character
        char_zoom = choice(SCALE)
        char_size = char.size
        new_width = int(char_size[0] * char_zoom)
        new_height = int(char_size[1] * char_zoom)

        #Resize Characters
        char = char.resize((new_width, new_height))
        background = background.resize(SIZE)
        size = background.size
        
        #random vị trí characters
        x = randrange(size[0] - char_size[0])
        y = randrange(size[1] - char_size[1])

        #Ghép characters vào background
        background.paste(char, (x,y) ,mask=char)

        #Write file .txt theo format yolov5
        w = char.size[0] - 25
        h = char.size[1] - 25

        x += w // 2
        y += h // 2

        x /= size[0]
        w /= size[0]

        y /= size[1]
        h /= size[1]

        name = id_generator()

        background.save(f"./{mode}_data/images/{name}.png")

        f = open(f"./{mode}_data/labels/{name}.txt", "x")
        f.write(f"{class_id} {x} {y} {w} {h}")
        f.close()



if __name__ == '__main__':
    build_data(1000, "train")