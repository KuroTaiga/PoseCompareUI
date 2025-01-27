# 标准库导入
import sys
import os
import json
import subprocess
from collections import defaultdict, Counter

# 第三方库导入
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
# 设置 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
def determine_posture(landmarks, joint_positions):
    """
    判断整体姿态（站立、坐姿、躺卧）
    """
    # 获取关键点的垂直位置关系
    hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + 
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + 
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
    ankle_y = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y + 
               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) / 2
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y

    # 计算关键点的水平位置
    hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + 
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2
    shoulder_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + 
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2

    # 计算躯干倾角
    torso_angle = calculate_angle(
        (shoulder_x, shoulder_y),
        (hip_x, hip_y),
        (hip_x, hip_y - 0.1)  # 创建一个垂直参考点
    )

    # 判断是否为躺卧姿势
    is_lying = abs(shoulder_y - hip_y) < 0.15 and abs(nose_y - hip_y) < 0.3

    # 判断是否为坐姿
    hip_to_ankle_dist = abs(hip_y - ankle_y)
    is_sitting = (hip_to_ankle_dist < 0.3 and  # 臀部和脚踝距离较近
                 hip_y > shoulder_y and         # 臀部高于肩部
                 abs(torso_angle - 90) < 30)    # 躯干接近垂直

    return {
        'is_lying': is_lying,
        'is_sitting': is_sitting,
        'torso_angle': torso_angle
    }

def determine_torso_position(posture, torso_angle):
    """
    基于姿态和躯干角度判断躯干位置
    """
    if posture['is_lying']:
        return "lying_flat"
    elif posture['is_sitting']:
        if torso_angle > 165:
            return "seated_upright"
        elif 135 < torso_angle <= 165:
            return "seated_leaned"
        else:
            return "seated_bent"
    else:  # 站立姿势
        if torso_angle > 165:
            return "upright"
        elif 135 < torso_angle <= 165:
            return "leaned_forward"
        else:
            return "bent"

def get_joint_positions_from_video(video_path, pose):
    """
    从视频路径提取关节位置,不进行平滑处理和异常值检测。
    """
    cap = cv2.VideoCapture(video_path)
    joint_positions_over_time = []
    frame_count = 0
    detected_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image)
        if results_pose.pose_landmarks:
            detected_frames += 1
            landmarks = results_pose.pose_landmarks.landmark
            joint_positions = {
                'left_shoulder': (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
                'right_shoulder': (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
                'left_elbow': (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
                'right_elbow': (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
                'left_wrist': (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
                'right_wrist': (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
                'left_hip': (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
                'right_hip': (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
                'left_knee': (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
                'right_knee': (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
                'left_ankle': (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
                'right_ankle': (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y),
                'nose': (landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                         landmarks[mp_pose.PoseLandmark.NOSE.value].y),
                'left_foot': (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y),
                'right_foot': (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y),
                'left_hand': (landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y),
                'right_hand': (landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y),
            }
            
            # 添加姿态判断
            posture = determine_posture(landmarks, joint_positions)
            
            # 更新躯干位置判断
            joint_positions['posture'] = determine_torso_position(posture, posture['torso_angle'])
            
            # 添加 spine、shoulder_center 和 hip_center 的计算
            joint_positions['spine'] = ((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                                        (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2)
            joint_positions['shoulder_center'] = joint_positions['spine']
            joint_positions['hip_center'] = ((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                                             (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2)
            
            joint_positions_over_time.append(joint_positions)
    cap.release()
    print(f"Total frames: {frame_count}, Detected frames: {detected_frames}")
    
    return joint_positions_over_time