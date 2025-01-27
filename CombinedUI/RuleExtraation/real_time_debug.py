import logging
import json
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import mediapipe as mp
import signal
import subprocess

# 修改路径设置
BASE_PATH = '/home/bizon/xiang/new_computer'  # 新的基础路径
SRC_PATH = os.path.join(BASE_PATH, 'src')
VIDEO_PATH = os.path.join(BASE_PATH, 'origin_test_video')
YOLO_PATH = os.path.join(BASE_PATH, 'yolov7')
YOLO_SCRIPT = os.path.join(YOLO_PATH, 'detect_revise.py')
YOLO_WEIGHTS = os.path.join(BASE_PATH, 'best.pt')
EXERCISE_JSON = os.path.join(BASE_PATH, 'exercise_generate_最新.json')
GENERATE_VIDEO_PATH = os.path.join(BASE_PATH, 'generate_video')

def verify_and_import():
    """验证所需路径和导入必要模块"""
    required_paths = {
        'Base Path': BASE_PATH,
        'Source Path': SRC_PATH,
        'Video Path': VIDEO_PATH,
        'YOLO Script': YOLO_SCRIPT,
        'YOLO Weights': YOLO_WEIGHTS,
        'Exercise JSON': EXERCISE_JSON,
        'Generate Video Path': GENERATE_VIDEO_PATH
    }

    for name, path in required_paths.items():
        if not os.path.exists(path):
            if name == 'Generate Video Path':
                print(f"创建输出视频目录: {path}")
                os.makedirs(path)
            else:
                raise Exception(f"Error: {name} not found at {path}")

    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)

    try:
        # 修改全局变量声明
        global generate_joint_features, detect_equipment_via_script
        from feature_extraction import generate_joint_features
        from equipment_detection import detect_equipment_via_script  # 确保这行正确导入
        print("Successfully imported required modules from src")
    except ImportError as e:
        print(f"Error importing modules: {str(e)}")
        raise

def get_joint_positions_from_video(video_path):
    """使用与pose_processing.py相同的MediaPipe实现"""
    try:
        # 初始化MediaPipe，保持与pose_processing.py相同的参数
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True
        )

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f"处理视频: {os.path.basename(video_path)}")
        print(f"总帧数: {total_frames}, FPS: {fps}")

        joint_positions_over_time = []
        frame_count = 0

        # MediaPipe关键点映射
        keypoint_mapping = {
            'nose': mp_pose.PoseLandmark.NOSE,
            'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE
        }

        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # 转换BGR到RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    joint_positions = {}
                    for name, landmark_id in mp_pose.PoseLandmark.__members__.items():
                        landmark = results.pose_landmarks.landmark[landmark_id]
                        joint_positions[name.lower()] = (landmark.x, landmark.y)
                    
                    # 添加计算合成关键点的代码
                    shoulder_center_x = (joint_positions['left_shoulder'][0] + joint_positions['right_shoulder'][0]) / 2
                    shoulder_center_y = (joint_positions['left_shoulder'][1] + joint_positions['right_shoulder'][1]) / 2
                    hip_center_x = (joint_positions['left_hip'][0] + joint_positions['right_hip'][0]) / 2
                    hip_center_y = (joint_positions['left_hip'][1] + joint_positions['right_hip'][1]) / 2

                    # 添加合成关键点
                    joint_positions.update({
                        'spine': (shoulder_center_x, shoulder_center_y),
                        'shoulder_center': (shoulder_center_x, shoulder_center_y),
                        'hip_center': (hip_center_x, hip_center_y),
                        'left_foot': joint_positions['left_ankle'],
                        'right_foot': joint_positions['right_ankle'],
                        'left_hand': joint_positions['left_wrist'],
                        'right_hand': joint_positions['right_wrist'],
                        'mid_hip': (hip_center_x, hip_center_y),
                        'mid_shoulder': (shoulder_center_x, shoulder_center_y),
                        'neck': (shoulder_center_x, shoulder_center_y)
                    })
                    
                    joint_positions_over_time.append(joint_positions)

                frame_count += 1
                pbar.update(1)

        cap.release()
        pose.close()

        print(f"成功从 {len(joint_positions_over_time)} 帧中提取姿态数据")
        return joint_positions_over_time

    except Exception as e:
        print(f"处理视频时出错: {str(e)}")
        return []

def convert_features_to_text(features):
    """将特征转换为文本描述，适配新的简化特征结构"""
    if not features:
        return ""
    
    text_parts = []
    
    # 添加设备信息
    if 'equipment' in features:
        text_parts.append(f"equipment:{features['equipment']}")
    
    # 添加简化后的动作状态
    for part in ['arm', 'body', 'leg']:
        if part in features:
            text_parts.append(f"{part}:{features[part]}")  # 直接使用 move/stale 状态
    
    return " ".join(text_parts)

def collect_video_files():
    """收集视频文件"""
    if not os.path.exists(VIDEO_PATH):
        raise Exception(f"Video directory not found: {VIDEO_PATH}")

    exercise_rules = build_exercise_rules()
    allowed_exercises = list(exercise_rules.keys())

    video_files = []
    exercise_names = []

    for file in os.listdir(VIDEO_PATH):
        if file.endswith(".mp4"):
            matched_exercise = next(
                (exercise for exercise in allowed_exercises
                 if exercise.lower() in file.lower()),
                None
            )

            if matched_exercise:
                video_path = os.path.join(VIDEO_PATH, file)
                video_files.append(video_path)
                exercise_names.append(matched_exercise)

    return video_files, exercise_names

def build_exercise_rules():
    """读取运动规则JSON文件"""
    try:
        with open(EXERCISE_JSON, 'r', encoding='utf-8') as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    except Exception as e:
        print(f"Error loading exercise.json: {e}")
        return OrderedDict()

def calculate_custom_similarity(video_vector, exercise_vectors, video_features, exercise_rules):
    """计算自定义相似度"""
    base_similarities = cosine_similarity(video_vector, exercise_vectors)[0]
    adjusted_similarities = base_similarities.copy()

    for i, exercise in enumerate(exercise_rules):
        rule = exercise_rules[exercise]
        score = 1.0

        # 检查关键特征匹配
        for key_feature in ["movement.primary", "pose.torso", "pose.arms.shoulder_position"]:
            if key_feature in rule and key_feature in video_features:
                if rule[key_feature] == video_features[key_feature]:
                    score *= 1.2
                else:
                    score *= 0.8

        # 检查设备匹配
        if rule.get("equipment", "") == video_features.get("equipment", ""):
            score *= 1.5
        else:
            score *= 0.5

        adjusted_similarities[i] *= score

    return adjusted_similarities

def apply_butterworth_filter(joint_positions_sequence, fps):
    try:
        from scipy import signal
        
        # 确保数据不为空
        if not joint_positions_sequence:
            return joint_positions_sequence
            
        # 创建一个新的序列来存储滤波后的数据
        filtered_sequence = []
        
        # 设置滤波器参数
        order = 4  # 滤波器阶数
        cutoff = 7.0  # 截止频率
        nyquist = fps / 2.0
        normalized_cutoff = cutoff / nyquist
        
        # 创建Butterworth滤波器
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        
        # 对每个关键点分别进行滤波
        for frame in range(len(joint_positions_sequence)):
            filtered_positions = {}
            
            # 获取第一个有效帧的关键点列表
            if frame == 0:
                keypoints = joint_positions_sequence[frame].keys()
            
            for keypoint in keypoints:
                # 收集该关键点在所有帧中的x和y坐标
                x_coords = []
                y_coords = []
                
                for positions in joint_positions_sequence:
                    if keypoint in positions and positions[keypoint] is not None:
                        x, y = positions[keypoint]
                        x_coords.append(x)
                        y_coords.append(y)
                    else:
                        # 如果某帧缺失该关键点，使用前一帧的值
                        if x_coords:
                            x_coords.append(x_coords[-1])
                            y_coords.append(y_coords[-1])
                        else:
                            x_coords.append(0.0)
                            y_coords.append(0.0)
                
                # 应用滤波器
                if len(x_coords) > order:
                    filtered_x = signal.filtfilt(b, a, x_coords)
                    filtered_y = signal.filtfilt(b, a, y_coords)
                    filtered_positions[keypoint] = (filtered_x[frame], filtered_y[frame])
                else:
                    # 如果数据点太少，保持原始值
                    filtered_positions[keypoint] = (x_coords[frame], y_coords[frame])
            
            filtered_sequence.append(filtered_positions)
        
        return filtered_sequence
        
    except Exception as e:
        print(f"滤波处理出错，使用原始数据: {str(e)}")
        return joint_positions_sequence

def process_video_direct(video_name):
    try:
        print(f"\n开始处理视频: {video_name}")
        
        # 设置路径
        input_video_path = os.path.join(VIDEO_PATH, video_name)
        base_name = os.path.splitext(video_name)[0]
        filtered_path = os.path.join(GENERATE_VIDEO_PATH, f"{base_name}_filtered.mp4")
        features_path = os.path.join(GENERATE_VIDEO_PATH, f"{base_name}_features.mp4")

        # 检测器材
        detected_equipment = detect_equipment_via_script(
            YOLO_SCRIPT,
            YOLO_WEIGHTS,
            input_video_path
        )
        print(f"检测到的设备: {detected_equipment}")

        # 获取视频信息
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 1. 使用 MediaPipe 处理视频并生成关节显示视频
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out_filtered = cv2.VideoWriter(filtered_path, fourcc, fps, (width, height))
        out_features = cv2.VideoWriter(features_path, fourcc, fps, (width, height))

        if not out_filtered.isOpened() or not out_features.isOpened():
            print("MP4V编码器失败，尝试MJPG...")
            filtered_path = filtered_path.replace('.mp4', '.avi')
            features_path = features_path.replace('.mp4', '.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_filtered = cv2.VideoWriter(filtered_path, fourcc, fps, (width, height))
            out_features = cv2.VideoWriter(features_path, fourcc, fps, (width, height))

        # 收集关节位置数据
        joint_positions_sequence = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 在处理每一帧的循环中添加角度计算和显示
        with tqdm(total=total_frames, desc="处理视频帧") as pbar:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # 处理帧
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # 生成关节显示视频
                filtered_frame = image.copy()
                if results.pose_landmarks:
                    # 绘制姿态标记
                    mp_drawing.draw_landmarks(
                        filtered_frame, 
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

                    # 计算并显示手臂角度
                    def calculate_angle(a, b, c):
                        a = np.array([a.x, a.y])
                        b = np.array([b.x, b.y])
                        c = np.array([c.x, c.y])
                        
                        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                        angle = np.abs(radians*180.0/np.pi)
                        
                        if angle > 180.0:
                            angle = 360-angle
                            
                        return angle

                    # 计算左手臂角度
                    left_arm_angle = calculate_angle(
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                    )

                    # 计算右手臂角度
                    right_arm_angle = calculate_angle(
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                    )

                    # 计算左腿角度
                    left_leg_angle = calculate_angle(
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                    )

                    # 计算右腿角度
                    right_leg_angle = calculate_angle(
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    )

                    # 在图像上显示角度（使用绿色，更容易看清）
                    cv2.putText(filtered_frame, f"L Arm: {left_arm_angle:.1f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 绿色
                    cv2.putText(filtered_frame, f"R Arm: {right_arm_angle:.1f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 绿色
                    cv2.putText(filtered_frame, f"L Leg: {left_leg_angle:.1f}", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 绿色
                    cv2.putText(filtered_frame, f"R Leg: {right_leg_angle:.1f}", 
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 绿色

                    # 收集关节位置
                    positions = {}
                    for name, landmark_id in mp_pose.PoseLandmark.__members__.items():
                        landmark = results.pose_landmarks.landmark[landmark_id]
                        positions[name.lower()] = (landmark.x, landmark.y)
                    
                    # 添加合成关键点
                    shoulder_center_x = (positions['left_shoulder'][0] + positions['right_shoulder'][0]) / 2
                    shoulder_center_y = (positions['left_shoulder'][1] + positions['right_shoulder'][1]) / 2
                    hip_center_x = (positions['left_hip'][0] + positions['right_hip'][0]) / 2
                    hip_center_y = (positions['left_hip'][1] + positions['right_hip'][1]) / 2
                    
                    positions.update({
                        'spine': (shoulder_center_x, shoulder_center_y),
                        'shoulder_center': (shoulder_center_x, shoulder_center_y),
                        'hip_center': (hip_center_x, hip_center_y),
                        'left_foot': positions['left_ankle'],
                        'right_foot': positions['right_ankle'],
                        'left_hand': positions['left_wrist'],
                        'right_hand': positions['right_wrist'],
                        'mid_hip': (hip_center_x, hip_center_y),
                        'mid_shoulder': (shoulder_center_x, shoulder_center_y),
                        'neck': (shoulder_center_x, shoulder_center_y)
                    })
                    
                    joint_positions_sequence.append(positions)
                else:
                    joint_positions_sequence.append(None)

                out_filtered.write(filtered_frame)
                pbar.update(1)

        # 生成整个视频的特征
        features = generate_joint_features(joint_positions_sequence, detected_equipment)
        filtered_features = filter_features(features)

        # 2. 生成特征显示视频
        feature_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if filtered_features:
            y_offset = 30
            line_height = 25
            
            # 显示设备
            if 'equipment' in filtered_features:
                cv2.putText(feature_frame, f"Equipment: {filtered_features['equipment']}", 
                          (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (255, 255, 255), 1)
                y_offset += line_height

            # 显示各部位的运动状态
            for part in ['arm', 'body', 'leg']:
                if part in filtered_features:
                    cv2.putText(feature_frame, f"{part.upper()}: {filtered_features[part]}", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (255, 255, 255), 1)
                    y_offset += line_height

        # 将相同的特征帧写入整个视频
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        with tqdm(total=total_frames, desc="生成特征视频") as pbar:
            for _ in range(total_frames):
                out_features.write(feature_frame)
                pbar.update(1)

        # 释放资源
        cap.release()
        out_filtered.release()
        out_features.release()
        pose.close()

        print("已生成视频:")
        print(f"姿态显示视频: {filtered_path}")
        print(f"特征信息视频: {features_path}")

        return [filtered_path, features_path]

    except Exception as e:
        print(f"\n处理 {video_name} 时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

# 在特征显示部分添加过滤函数
def filter_features(features):
    """过滤掉空的特征，适配新的简化特征结构"""
    if not features:
        return {}
        
    filtered = {}
    
    # 处理equipment
    if 'equipment' in features and features['equipment']:
        filtered['equipment'] = features['equipment']
    
    # 处理简化后的三个主要类别：arm, body, leg
    for part in ['arm', 'body', 'leg']:
        if part in features:
            filtered[part] = features[part]  # 直接保存 move/stale 状态
    
    return filtered

if __name__ == "__main__":
    try:
        print("Starting program...")
        verify_and_import()
        
        # 确保输出目录存在
        if not os.path.exists(GENERATE_VIDEO_PATH):
            os.makedirs(GENERATE_VIDEO_PATH)
            print(f"创建输出目录: {GENERATE_VIDEO_PATH}")
        
        # 获取所有视频文件
        video_files = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]
        print(f"\n找到 {len(video_files)} 个视频文件")
        
        # 处理所有视频
        successful = 0
        failed = 0
        skipped = 0
        
        for video in video_files:
            output_path = process_video_direct(video)
            if output_path:
                successful += 1
            elif output_path is None and os.path.exists(os.path.join(GENERATE_VIDEO_PATH, f"processed_{video}")):
                skipped += 1
            else:
                failed += 1
        
        print("\n处理完成!")
        print(f"成功: {successful}")
        print(f"失败: {failed}")
        print(f"跳过: {skipped}")
        print(f"输出目录: {GENERATE_VIDEO_PATH}")
                
    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback
        print(traceback.format_exc())