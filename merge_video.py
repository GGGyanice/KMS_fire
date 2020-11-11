from moviepy.editor import *
import os
from natsort import natsorted


L = []

# 访问 video 文件夹
for root, dirs, files in os.walk("E:/kassie/火焰视频"):
    # 按文件名排序
    files.sort()
    # files = natsorted(files)
    # 遍历所有文件
    for file in files:
        # 如果后缀名为 .mp4
        if os.path.splitext(file)[1] == '.mp4':
            # 拼接成完整路径
            filePath = os.path.join(root, file)
            # 载入视频
            video = VideoFileClip(filePath)
            # 添加到数组
            L.append(video)

# 拼接
final_clip = concatenate_videoclips(L)

# 生成
final_clip.to_videofile("E:/kassie/火焰视频/target.mp4", fps=24, remove_temp=False)
