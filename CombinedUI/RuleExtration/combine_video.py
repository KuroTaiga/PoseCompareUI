import cv2
import os
import numpy as np
from tqdm import tqdm

def combine_videos(original_path, filtered_path, features_path, output_path):
    """
    将三个视频水平并排合并，使用 MP4 格式
    """
    try:
        # 打开所有视频
        cap_original = cv2.VideoCapture(original_path)
        cap_filtered = cv2.VideoCapture(filtered_path)
        cap_features = cv2.VideoCapture(features_path)
        
        # 检查是否所有视频都成功打开
        if not all([cap_original.isOpened(), cap_filtered.isOpened(), cap_features.isOpened()]):
            raise Exception("无法打开一个或多个视频文件")
            
        # 获取视频信息
        width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap_original.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建输出视频写入器 - 使用 MP4V 编码器
        combined_width = width * 3  # 三个视频并排
        
        # 确保输出路径使用 .mp4 扩展名
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        
        # 尝试不同的 MP4 兼容编码器
        codecs = [
            ('mp4v', '.mp4'),   # 标准 MP4 编码器
            ('avc1', '.mp4'),   # H.264 编码器
            ('H264', '.mp4'),   # 另一个 H.264 变体
        ]
        
        # 尝试不同的编码器
        success = False
        out = None
        
        for codec, ext in codecs:
            try:
                print(f"尝试使用 {codec} 编码器...")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                output_file = output_path.rsplit('.', 1)[0] + ext
                
                out = cv2.VideoWriter(output_file, fourcc, fps, (combined_width, height))
                
                if out.isOpened():
                    success = True
                    print(f"成功使用 {codec} 编码器")
                    output_path = output_file  # 更新输出路径
                    break
                else:
                    if out:
                        out.release()
            except Exception as e:
                print(f"{codec} 编码器失败: {str(e)}")
                if out:
                    out.release()
                continue
        
        if not success:
            raise Exception("没有可用的视频编码器")
            
        print(f"\n开始合并视频...")
        print(f"输入视频尺寸: {width}x{height}")
        print(f"输出视频尺寸: {combined_width}x{height}")
        print(f"帧率: {fps}")
        print(f"总帧数: {total_frames}")
        
        # 添加标题文本的函数
        def add_title(frame, title):
            """在视频帧上添加标题"""
            height, width = frame.shape[:2]
            
            # 计算右下角的位置
            text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x = width - text_size[0] - 10  # 距离右边缘10像素
            y = height - 20  # 距离底边20像素
            
            # 添加半透明黑色背景以提高文字可读性
            overlay = frame.copy()
            bg_x1 = x - 5
            bg_y1 = y - text_size[1] - 5
            bg_x2 = width - 5
            bg_y2 = height - 5
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            
            # 设置透明度
            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # 根据背景亮度选择文字颜色
            if title == "Original":
                color = (50, 205, 50)  # 浅绿色
            elif title == "Filtered Pose":
                color = (255, 165, 0)  # 橙色
            else:  # Features
                color = (147, 112, 219)  # 紫色
            
            # 添加文字
            cv2.putText(frame, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)
            
            return frame
        
        with tqdm(total=total_frames, desc="合并进度") as pbar:
            while True:
                # 读取每个视频的帧
                ret1, frame1 = cap_original.read()
                ret2, frame2 = cap_filtered.read()
                ret3, frame3 = cap_features.read()
                
                if not all([ret1, ret2, ret3]):
                    break
                
                # 添加标题
                frame1 = add_title(frame1, "Original")
                frame2 = add_title(frame2, "Filtered Pose")
                frame3 = add_title(frame3, "Features")
                
                # 水平合并帧
                combined_frame = np.hstack((frame1, frame2, frame3))
                
                # 写入合并后的帧
                out.write(combined_frame)
                
                pbar.update(1)
        
        # 释放资源
        cap_original.release()
        cap_filtered.release()
        cap_features.release()
        out.release()
        
        print(f"\n视频合并完成！")
        print(f"输出文件: {output_path}")
        
    except Exception as e:
        print(f"合并视频时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # 确保释放所有资源
        for cap in [cap_original, cap_filtered, cap_features, out]:
            if 'cap' in locals() and cap is not None:
                cap.release()

def process_all_videos(video_dir, generate_dir, output_base_dir):
    """
    处理目录中的所有视频组
    """
    try:
        # 创建输出目录
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"创建输出目录: {output_base_dir}")

        # 获取所有原始视频
        original_videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"\n找到 {len(original_videos)} 个原始视频")

        for video in original_videos:
            base_name = os.path.splitext(video)[0]

            # 构建文件路径 - 使用 .mp4
            original_path = os.path.join(video_dir, video)
            filtered_path = os.path.join(generate_dir, f"{base_name}_filtered.mp4")
            features_path = os.path.join(generate_dir, f"{base_name}_features.mp4")

            # 打印详细的文件路径信息
            print(f"\n处理视频: {video}")
            print(f"检查文件是否存在:")
            print(f"原始视频: {original_path} - {os.path.exists(original_path)}")
            print(f"过滤视频: {filtered_path} - {os.path.exists(filtered_path)}")
            print(f"特征视频: {features_path} - {os.path.exists(features_path)}")

            # 检查所需文件是否都存在
            if not all([os.path.exists(p) for p in [original_path, filtered_path, features_path]]):
                print(f"跳过 {video} - 缺少必要的输入文件")
                continue

            output_path = os.path.join(output_base_dir, f"{base_name}_combined.mp4")
            combine_videos(original_path, filtered_path, features_path, output_path)

    except Exception as e:
        print(f"处理视频目录时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # 更新基础路径为新电脑的路径
    BASE_PATH = os.getcwd()
    VIDEO_PATH = os.path.join(BASE_PATH, 'origin_test_video')
    GENERATE_VIDEO_PATH = os.path.join(BASE_PATH, 'generate_video')
    COMBINED_VIDEO_PATH = os.path.join(BASE_PATH, 'combined_video')

    print("开始处理视频...")
    process_all_videos(VIDEO_PATH, GENERATE_VIDEO_PATH, COMBINED_VIDEO_PATH)
    print("\n所有视频处理完成!")