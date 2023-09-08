import cv2
import os

"""将代码输出的测试文件名字修改为原始的名字"""

def file_change(file):
    print('file_name={}'.format(file))
    file_tail = '.png'
    for sub_name in os.listdir(file):
        # if file.split('/')[-2] == 'gtFine':
        #     file_tail = 'gtFine_labelIds.png'
        # else:
        #     file_tail = 'leftImg8bit.png'
        sub_file_name = file + sub_name
        for sub2_name in os.listdir(sub_file_name):
            print('sub_file_name={}'.format(sub_file_name))
            sub2_file_name = sub_file_name+'/'+sub2_name
            sub2_name_new = 'frame' + str(sub2_name.split('_')[1]) + file_tail
            sub2_file_name_new = sub_file_name+'/'+sub2_name_new
            os.rename(sub2_file_name, sub2_file_name_new)
        sub_name_new = sub_name[:3] + '_' + sub_name[3:4]
        sub_file_name_new = file + sub_name_new
        os.rename(sub_file_name, sub_file_name_new)

if __name__ == '__main__':

    org_file = r"E:/SurgicalRobot/endovis2018_orgcode/hr_tma_out/"
    file_change(org_file)
    print('over over')