import pandas as pd
import cv2
import numpy as np
import os
import shutil

# 读取数据
path = 'dataset/fer2013.csv'
df = pd.read_csv(path)

# 提取emotion数据
df_y = df[['emotion']]
# 提取pixels数据
df_x = df[['pixels']]

# 将emotion写入emotion.csv
df_y.to_csv('dataset/emotion.csv', index=False, header=False)
# 将pixels数据写入pixels.csv
df_x.to_csv('dataset/pixels.csv', index=False, header=False)

# 指定存放图片的路径
image_save_path = 'face_images'
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

# 读取像素数据
data = df_x['pixels'].values

# 按行取数据并生成图片
for i, pixel_data in enumerate(data):
    face_array = np.array(pixel_data.split(), dtype='uint8').reshape((48, 48))  # reshape
    cv2.imwrite(os.path.join(image_save_path, '{}.jpg'.format(i)), face_array)  # 写图片

# 创建train_set和verify_set文件夹
train_set_path = os.path.join(image_save_path, 'train_set')
verify_set_path = os.path.join(image_save_path, 'verify_set')
os.makedirs(train_set_path, exist_ok=True)
os.makedirs(verify_set_path, exist_ok=True)

# 获取所有图片文件名
image_files = [f for f in os.listdir(image_save_path) if f.endswith('.jpg')]
total_images = len(image_files)
train_images_count = int(total_images * 0.7)

# 按比例分配图片到train_set和verify_set
for i, image_file in enumerate(image_files):
    src_path = os.path.join(image_save_path, image_file)
    if i < train_images_count:
        dest_path = os.path.join(train_set_path, image_file)
    else:
        dest_path = os.path.join(verify_set_path, image_file)
    shutil.move(src_path, dest_path)

def image_emotion_mapping(path):
    # 读取emotion文件
    df_emotion = pd.read_csv('dataset/emotion.csv', header=None)
    # 查看该文件夹下所有文件
    files_dir = os.listdir(path)
    # 用于存放图片名
    path_list = []
    # 用于存放图片对应的emotion
    emotion_list = []
    # 遍历该文件夹下的所有文件
    for file_dir in files_dir:
        # 如果某文件是图片，则将其文件名以及对应的emotion取出，分别放入path_list和emotion_list这两个列表中
        if os.path.splitext(file_dir)[1] == ".jpg":
            path_list.append(file_dir)
            index = int(os.path.splitext(file_dir)[0])
            emotion_list.append(df_emotion.iat[index, 0])

    # 将两个列表写进image_emotion.csv文件
    path_s = pd.Series(path_list)
    emotion_s = pd.Series(emotion_list)
    df = pd.DataFrame()
    df['path'] = path_s
    df['emotion'] = emotion_s
    df.to_csv(os.path.join(path, 'image_emotion.csv'), index=False, header=False)

def main():
    # 为train_set和verify_set生成image_emotion.csv文件
    image_emotion_mapping(train_set_path)
    image_emotion_mapping(verify_set_path)

if __name__ == "__main__":
    main()
