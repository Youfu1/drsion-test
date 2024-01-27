import cv2
import os
import glob
"""
将视频size处理成224*224的图像帧并按照原来的目录结构保存
"""

def preprocess_frame(frame, target_size=(224, 224)):
    """
    对单个视频帧进行预处理。
    """
    frame_resized = cv2.resize(frame, target_size)
    return frame_resized

def save_frames_from_video(video_path, output_dir, video_name ,target_size=(224, 224)):
    """
    从视频中提取帧并保存为图像。
    """
    #！！！这里可以更改视频来源 可以让我们的外设相机视频流入
    #cap = cv2.VideoCapture(0)将开始从系统上的第一个摄像头（如网络摄像头）捕获视频
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame, target_size)
        save_path = os.path.join(output_dir, f'{video_name}_{frame_count}.jpg')
        cv2.imwrite(save_path, processed_frame)
        frame_count += 1

    cap.release()

def process_videos_in_directory(root_dir, output_root_dir, target_size=(224, 224)):
    """
    遍历指定目录中的所有视频文件，并将每个视频的帧保存为图像。
    """
    for subdir, dirs, files in os.walk(root_dir):
        for file in glob.glob(os.path.join(subdir, '*.avi')):
            # 创建输出目录，保持原始目录结构
            relative_path = os.path.relpath(subdir, root_dir)
            output_dir = os.path.join(output_root_dir, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            video_name = os.path.basename(file)
            save_frames_from_video(file, output_dir, video_name, target_size)

# 根目录路径
root_dir = 'UCF_copy'

# 输出根目录路径
output_root_dir = 'UCF_pro'

# 处理 test 和 train 目录
process_videos_in_directory(os.path.join(root_dir, 'test'), os.path.join(output_root_dir,'test'))
process_videos_in_directory(os.path.join(root_dir, 'train'), os.path.join(output_root_dir,'train'))
