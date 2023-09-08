import glob
import os.path

from PIL import Image
import cv2

CLASSES = ("Background", "Tissue Space", "Ultrasonic Scalpel", "Bipolar Forceps",
           "Intestinal Forceps", "Clip Applier", "Cutting and Closing Instrument", "Suction Instrument")
class_list = [0,0,0,0,0,0,0,0]
# root_dir = '/data2/hyb/DataSurgery/Lapavis_train2250_test750/gtFine/train/'
root_dir = '/data2/hyb/DataSurgery/Lapavis_train2250_test750/gtFine/test/'
dataset = glob.glob(os.path.join(root_dir,'*.png'))
for data_path in dataset:
    image = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
    if 1 in image:
        class_list[1] = class_list[1] + 1
    if 2 in image:
        class_list[2] = class_list[2] + 1
    if 3 in image:
        class_list[3] = class_list[3] + 1
    if 4 in image:
        class_list[4] = class_list[4] + 1
    if 5 in image:
        class_list[5] = class_list[5] + 1
    if 6 in image:
        class_list[6] = class_list[6] + 1
    if 7 in image:
        class_list[7] = class_list[7] + 1
    else:
        continue
print(1)