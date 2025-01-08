import gradio as gr
import cv2
import torch
import numpy as np
import mediapipe as mp
# from mmcv import Config
# from mmpose.models import build_posenet
# from mmcv.runner import load_checkpoint
import torchvision.transforms as transforms
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from FDHumans.hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from FDHumans.hmr2.utils import recursive_to
import subprocess
import os
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Vitpose is decapitated
'''
class VitPoseWrapper:
    """Wrapper for ViTPose model."""
    def __init__(self, config_path, weights_path, input_size=(192, 256), device=None):
        self.input_width, self.input_height = input_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # Load configuration and model
            cfg = Config.fromfile(config_path)
            self.model = build_posenet(cfg.model)
            load_checkpoint(self.model, weights_path, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()

            # Transformation pipeline
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # Keypoint connections (COCO format)
            self.keypoint_connections = [
                (15, 13), (13, 11), (16, 14), (14, 12),  # Limbs
                (11, 12), (5, 11), (6, 12), (5, 6),      # Hips to shoulders
                (5, 7), (6, 8), (7, 9), (8, 10),        # Neck to arms
                (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # Face and shoulders
            ]

            logging.info("ViTPose model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize ViTPose model: {e}", exc_info=True)
            raise

    def preprocess_frame(self, frame):
        try:
            original_height, original_width = frame.shape[:2]
            frame_resized = cv2.resize(frame, (self.input_width, self.input_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

            img_metas = [{
                'img_shape': (self.input_height, self.input_width, 3),
                'original_shape': (original_height, original_width, 3),
                'scale_factor': np.array([
                    original_width / self.input_width,
                    original_height / self.input_height
                ]),
                'center': np.array([self.input_width // 2, self.input_height // 2]),
                'scale': np.array([1.0, 1.0]),
                'rotation': 0,
                'flip_pairs': None,
                'dataset_idx': 0,
                'image_file': None
            }]
            return input_tensor, img_metas
        except Exception as e:
            logging.error(f"Error preprocessing frame: {e}", exc_info=True)
            return None, None

    def process_frame(self, frame, show_background=True):
        try:
            input_tensor, img_metas = self.preprocess_frame(frame)
            if input_tensor is None:
                return frame

            display_frame = frame.copy() if show_background else np.zeros(frame.shape, dtype=np.uint8)

            with torch.no_grad():
                output = self.model(img=input_tensor, img_metas=img_metas, return_loss=False)
                keypoints = output['preds'][0]

            if keypoints is not None and len(keypoints) > 0:
                meta = img_metas[0]
                scale_factor = meta['scale_factor']

                # Scale keypoints back to original frame size
                original_keypoints = []
                for kp in keypoints:
                    orig_x = int(kp[0] * scale_factor[0])
                    orig_y = int(kp[1] * scale_factor[1])
                    original_keypoints.append((orig_x, orig_y, kp[2]))

                # Draw skeleton
                confidence_threshold = 0.3
                for connection in self.keypoint_connections:
                    pt1_idx, pt2_idx = connection
                    if (original_keypoints[pt1_idx][2] > confidence_threshold and
                            original_keypoints[pt2_idx][2] > confidence_threshold):
                        pt1 = tuple(map(int, original_keypoints[pt1_idx][:2]))
                        pt2 = tuple(map(int, original_keypoints[pt2_idx][:2]))
                        color = (255, 255, 255) if not show_background else (0, 255, 0)
                        cv2.line(display_frame, pt1, pt2, color, 2)

                # Draw keypoints
                for x, y, conf in original_keypoints:
                    if conf > confidence_threshold:
                        color = (255, 255, 255) if not show_background else (255, 0, 0)
                        cv2.circle(display_frame, (int(x), int(y)), 4, color, -1)

            return display_frame
        except Exception as e:
            logging.error(f"Error processing frame with ViTPose: {e}", exc_info=True)
            return frame
'''
class FourDHumanWrapper:
    """Wrapper for 4DHuman model."""
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT

        try:
            self.model, self.model_cfg = load_hmr2(self.checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
            logging.info("4DHuman model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize 4DHuman model: {e}", exc_info=True)
            raise

    def preprocess_frame(self, frame):
        try:
            frame_resized = cv2.resize(frame, (256, 256))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
            input_tensor = torch.tensor(frame_rgb.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
            return input_tensor.to(self.device)
        except Exception as e:
            logging.error(f"Error preprocessing frame: {e}", exc_info=True)
            return None

    def process_frame(self, frame, show_background=True):
        try:
            display_frame = frame.copy() if show_background else np.zeros(frame.shape, dtype=np.uint8)
            input_tensor = self.preprocess_frame(frame)
            
            if input_tensor is not None:
                with torch.no_grad():
                    batch = {"img": input_tensor}
                    batch = recursive_to(batch, self.device)
                    output = self.model(batch)
                    
                    vertices = output["pred_vertices"][0].cpu().numpy()
                    camera_params = output["pred_cam"][0].cpu().numpy()
                    
                    display_frame = self.render_mesh(display_frame, vertices, camera_params)
            
            return display_frame
        except Exception as e:
            logging.error(f"Error processing frame with 4DHuman model: {e}", exc_info=True)
            return frame

    def render_mesh(self, frame, vertices, camera_params):
        try:
            s, tx, ty = camera_params
            img_h, img_w = frame.shape[:2]
            
            projected_vertices = vertices[:, :2] * s + np.array([tx, ty])
            projected_vertices[:, 0] = (projected_vertices[:, 0] + 1) * img_w / 2.0
            projected_vertices[:, 1] = img_h-(1-projected_vertices[:, 1]) * img_h / 2.0

            for v in projected_vertices.astype(int):
                cv2.circle(frame, tuple(v), 2, (0, 255, 0), -1)
            
            return frame
        except Exception as e:
            logging.error(f"Error rendering mesh: {e}", exc_info=True)
            return frame
        
    def process_video(self, video_path, output_path,show_background = True):
        try:
            # temp_output_path = f'temp_{output_path}'
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # video_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            frame_count = 0 
            max_frames = 300  # Limit processing to 100 frames

            # First read all frames and landmarks
            while cap.isOpened() and frame_count < max_frames:
            # while cap.isOpened():
                ret,frame = cap.read()
                if not ret:
                    break
                frame_count+=1
                print(f"\rProcessing frame {frame_count}", end="")

                processed_frame = self.process_frame(frame,show_background)
                video_writer.write(processed_frame)
            cap.release()
            video_writer.release()
            # ffmpeg_command = [
            #     "ffmpeg", "-y", "-i", temp_output_path,
            #     "-vcodec", "libx264", "-acodec", "aac", output_path
            # ]
            # try:
            #     subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # except subprocess.CalledProcessError as e:
            #     print(f"FFmpeg error:\n{e.stderr.decode()}")
            #     raise

            logging.info(f"Video processing complete. Output saved to {output_path}")
            return True
        except Exception as e:
            logging.error(f"Error processing video: {e}", exc_info=True)
            return False

class PoseEstimationApp:
    def __init__(self):
        self.vitpose_config_path = 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_large_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py'
        self.vitpose_weights_path = './checkpoints/vitpose_checkpoint.pth'
        
        # Initialize models

        # try:
        #     self.vitpose = VitPoseWrapper(self.vitpose_config_path, self.vitpose_weights_path)
        # except Exception as e:
        #     logging.error(f"Failed to initialize VitPose: {e}")
        #     self.vitpose = None

        try:
            self.fdh_model = FourDHumanWrapper()
        except Exception as e:
            logging.error(f"Failed to initialize 4DHuman: {e}")
            self.fdh_model = None

        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mediapipe_pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5
            )
        except Exception as e:
            logging.error(f"Failed to initialize MediaPipe: {e}")
            self.mediapipe_pose = None

    def process_mediapipe(self, frame, show_background=True):
        try:
            if self.mediapipe_pose:
                display_frame = frame.copy() if show_background else np.zeros(frame.shape, dtype=np.uint8)
                results = self.mediapipe_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    color = (255, 255, 255) if not show_background else None
                    drawing_spec = self.mp_drawing.DrawingSpec(
                        color=color,
                        thickness=2,
                        circle_radius=2
                    ) if not show_background else None
                    
                    self.mp_drawing.draw_landmarks(
                        display_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )
                return display_frame
        except Exception as e:
            logging.error(f"Error in MediaPipe processing: {e}")
        return frame

    def process_video(self, video_path, selected_models, show_background=True):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frames = []
                if "Original" in selected_models:
                    processed_frames.append(frame)
                    
                if "MediaPipe" in selected_models and self.mediapipe_pose:
                    processed_frames.append(self.process_mediapipe(frame, show_background))
                    
                # if "ViTPose" in selected_models and self.vitpose:
                #     processed_frames.append(self.vitpose.process_frame(frame, show_background))
                    
                if "4DHuman" in selected_models and self.fdh_model:
                    processed_frames.append(self.fdh_model.process_frame(frame, show_background))
                
                # Combine frames horizontally
                if processed_frames:
                    combined_frame = np.hstack(processed_frames)
                    frames.append(combined_frame)
                
            cap.release()
            return frames
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            return None

def create_gradio_interface():
    app = PoseEstimationApp()

    def process_video_file(video_path, selected_models, show_background):
        if not selected_models:
            return None
        frames = app.process_video(video_path, selected_models, show_background)
        if frames:
            return frames
        return None

    iface = gr.Interface(
        fn=process_video_file,
        inputs=[
            gr.Video(label="Upload Video"),
            gr.CheckboxGroup(
                choices=["Original", "MediaPipe", "ViTPose", "4DHuman"],
                label="Select Models",
                value=["Original"]
            ),
            gr.Checkbox(label="Show Background", value=True)
        ],
        outputs=gr.Video(label="Processed Video"),
        title="Pose Estimation Comparison",
        description="Compare different pose estimation models on video input.",
        examples=[],
        cache_examples=False
    )
    return iface

if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch()