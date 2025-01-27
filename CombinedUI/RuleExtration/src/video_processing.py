# video_processing.py

import os
import mediapipe as mp
from typing import List, Tuple

# 导入其他必要的模块
from feature_extraction import generate_joint_features, aggregate_features
from equipment_detection import detect_equipment_via_script
from pose_estimation import get_joint_positions_from_video

# 设置 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_video_for_exercise(video_path: str, detect_script_path: str, yolov7_weights_path: str):
    """
    处理单个视频，提取特征。
    """
    detected_equipment = detect_equipment_via_script(
        detect_script_path, yolov7_weights_path, video_path)
    
    joint_positions_over_time = get_joint_positions_from_video(video_path, pose)
    
    if not joint_positions_over_time:
        return None

    features_over_time = []
    previous_joint_positions = None
    for jp in joint_positions_over_time:
        features = generate_joint_features(jp, detected_equipment, previous_joint_positions)
        if features is not None:
            features_over_time.append(features)
            previous_joint_positions = jp

    if not features_over_time:
        return None

    return aggregate_features(features_over_time)

def collect_video_files(root_dir: str, allowed_exercises: List[str]) -> Tuple[List[str], List[str]]:
    video_files = []
    exercise_names = []
    
    for folder_num in range(1, 48):
        folder_name = f"folder_{folder_num}"
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".mp4"):
                file_name_without_ext = os.path.splitext(file)[0]
                
                matched_exercise = next((exercise for exercise in allowed_exercises if exercise.lower() in file_name_without_ext.lower()), None)
                
                if matched_exercise:
                    video_path = os.path.join(folder_path, file)
                    video_files.append(video_path)
                    exercise_names.append(matched_exercise)
    
    return video_files, exercise_names