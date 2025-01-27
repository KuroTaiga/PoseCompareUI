import logging
import cv2
import mediapipe as mp
import numpy as np
import sys

# 初始化 Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """计算三点之间的角度"""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        return angle
    except Exception as e:
        logging.warning(f"计算角度时出错: {str(e)}")
        return 0

def smooth_angles(angles, window_size=5, max_jump=30):
    """平滑角度数据，去除突变和抖动"""
    # 确保输入是 numpy 数组
    if not isinstance(angles, np.ndarray):
        angles = np.array(angles)
    
    if len(angles) < window_size:  # 使用 len() 替代条件判断
        return angles
    
    smoothed = np.copy(angles)
    
    # 处理突变值
    for i in range(1, len(angles)):
        jump = abs(angles[i] - angles[i-1])
        if jump > max_jump:
            # 如果角度突变太大，使用前一个值
            smoothed[i] = smoothed[i-1]
    
    # 使用滑动窗口平均值平滑数据
    for i in range(len(smoothed)):
        start = max(0, i - window_size // 2)
        end = min(len(smoothed), i + window_size // 2 + 1)
        smoothed[i] = np.mean(smoothed[start:end])
    
    return smoothed

def generate_joint_features(joint_positions, equipment):
    """基于角度变化范围判断运动状态"""
    features = {
        "equipment": equipment.lower(),
        "arm": "stale",
        "body": "stale",
        "leg": "stale"
    }
    
    if not joint_positions or len(joint_positions) < 2:
        return features
        
    # 收集所有帧的角度数据
    left_arm_angles = []   # 左手臂角度序列
    right_arm_angles = []  # 右手臂角度序列
    left_leg_angles = []   # 左腿角度序列
    right_leg_angles = []  # 右腿角度序列
    body_angles = []       # 躯干角度序列
    
    total_frames = len(joint_positions)
    left_arm_detected = 0  # 记录左手臂被检测到的帧数
    right_arm_detected = 0 # 记录右手臂被检测到的帧数
    
    for positions in joint_positions:
        if not positions:
            continue
            
        try:
            # 计算左手臂角度
            if all(k in positions for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                left_arm = calculate_angle(
                    positions['left_shoulder'],
                    positions['left_elbow'],
                    positions['left_wrist']
                )
                left_arm_angles.append(left_arm)
                left_arm_detected += 1
            
            # 计算右手臂角度
            if all(k in positions for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                right_arm = calculate_angle(
                    positions['right_shoulder'],
                    positions['right_elbow'],
                    positions['right_wrist']
                )
                right_arm_angles.append(right_arm)
                right_arm_detected += 1
            
            # 计算左腿角度
            left_leg = calculate_angle(
                positions['left_hip'],
                positions['left_knee'],
                positions['left_ankle']
            )
            left_leg_angles.append(left_leg)
            
            # 计算右腿角度
            right_leg = calculate_angle(
                positions['right_hip'],
                positions['right_knee'],
                positions['right_ankle']
            )
            right_leg_angles.append(right_leg)
            
            # 计算躯干角度（相对于垂直线）
            spine_top = (positions['shoulder_center'][0], positions['shoulder_center'][1] - 0.5)
            body_angle = calculate_angle(
                spine_top,
                positions['shoulder_center'],
                positions['hip_center']
            )
            body_angles.append(body_angle)
            
        except Exception as e:
            logging.warning(f"处理帧时出错: {str(e)}")
            continue
    
    def check_angle_range(angles, detected_frames, total_frames, threshold=20, min_detection_ratio=0.3):
        """检查角度序列的变化范围是否超过阈值"""
        # 修改空值检查的方式
        if not isinstance(angles, np.ndarray):
            angles = np.array(angles)
        if len(angles) == 0:  # 使用 len() 替代 not angles
            return False
            
        # 计算检测率
        detection_ratio = detected_frames / total_frames
        
        # 如果检测率太低，认为数据不可靠
        if detection_ratio < min_detection_ratio:
            logging.warning(f"检测率太低: {detection_ratio:.2f}, 检测到 {detected_frames}/{total_frames} 帧")
            return False
            
        # 平滑角度数据
        smoothed_angles = smooth_angles(angles)
        
        if len(smoothed_angles) < 3:  # 至少需要3个点才能判断
            return False
            
        # 计算平滑后的角度范围
        angle_range = np.max(smoothed_angles) - np.min(smoothed_angles)
        
        # 计算角度变化的稳定性
        angle_changes = np.abs(np.diff(smoothed_angles))
        stable_changes = np.mean(angle_changes) < 5  # 平均变化小于5度认为是稳定的
        
        # 打印调试信息
        logging.info(f"角度范围: {angle_range:.2f}, 检测率: {detection_ratio:.2f}, 平均变化: {np.mean(angle_changes):.2f}")
        
        # 只有当角度范围超过阈值且变化相对稳定时才认为是运动
        return angle_range > threshold and stable_changes
    
    # 对收集到的角度序列进行平滑处理
    left_arm_angles = smooth_angles(left_arm_angles) if left_arm_angles else []
    right_arm_angles = smooth_angles(right_arm_angles) if right_arm_angles else []
    left_leg_angles = smooth_angles(left_leg_angles) if left_leg_angles else []
    right_leg_angles = smooth_angles(right_leg_angles) if right_leg_angles else []
    body_angles = smooth_angles(body_angles) if body_angles else []
    
    # 分别判断左右手臂的运动状态
    left_arm_move = check_angle_range(left_arm_angles, left_arm_detected, total_frames)
    right_arm_move = check_angle_range(right_arm_angles, right_arm_detected, total_frames)
    
    # 如果一侧检测率高且显示运动，就认为是运动状态
    # 如果两侧检测率都高，需要两侧都显示静止才认为是静止状态
    if left_arm_detected/total_frames >= 0.3 and right_arm_detected/total_frames >= 0.3:
        features["arm"] = "move" if (left_arm_move or right_arm_move) else "stale"
    elif left_arm_detected/total_frames >= 0.3:
        features["arm"] = "move" if left_arm_move else "stale"
    elif right_arm_detected/total_frames >= 0.3:
        features["arm"] = "move" if right_arm_move else "stale"
    else:
        features["arm"] = "stale"  # 如果两侧检测率都太低，默认为静止
    
    # 分别判断左右腿的运动状态
    left_leg_move = check_angle_range(left_leg_angles, left_arm_detected, total_frames)
    right_leg_move = check_angle_range(right_leg_angles, right_arm_detected, total_frames)
    features["leg"] = "move" if (left_leg_move or right_leg_move) else "stale"
    
    # 判断躯干运动状态
    features["body"] = "move" if check_angle_range(body_angles, left_arm_detected, total_frames, threshold=15) else "stale"
    
    return features

def filter_features(features):
    """过滤掉空的特征"""
    if not features:
        return {}
        
    filtered = {}
    
    if 'equipment' in features and features['equipment']:
        filtered['equipment'] = features['equipment']
    
    for part in ['arm', 'body', 'leg']:
        if part in features:
            filtered[part] = features[part]
    
    return filtered

def aggregate_features(features_over_time):
    """
    聚合整个视频的特征
    """
    if not features_over_time:
        return {}

    # 统计每个部位的运动状态
    movement_stats = {
        "arm": {"move": 0, "stale": 0},
        "body": {"move": 0, "stale": 0},
        "leg": {"move": 0, "stale": 0}
    }

    # 计算每个状态的出现次数
    for features in features_over_time:
        for part in ["arm", "body", "leg"]:
            if part in features:
                movement_stats[part][features[part]] += 1

    # 根据统计结果决定最终状态
    final_features = {
        "equipment": features_over_time[-1]["equipment"]
    }

    # 如果超过20%的帧显示运动，就认为是运动状态
    threshold = len(features_over_time) * 0.2
    for part in ["arm", "body", "leg"]:
        final_features[part] = "move" if movement_stats[part]["move"] > threshold else "stale"

    return final_features