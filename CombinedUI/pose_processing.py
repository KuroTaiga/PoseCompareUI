import cv2
import mediapipe as mp
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d, splrep, splev, griddata
from filterpy.kalman import KalmanFilter
from pykrige.ok import OrdinaryKriging
from mediapipe.framework.formats import landmark_pb2
import subprocess
import os

class PoseProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_video(self, input_video_path, method='original'):
        print(f"Start processing with {method} method...")
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output video file
        # temp_output_path = f'temp_output_{method}.mp4'
        output_path = f'output_{method}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Store landmarks for all frames
        all_landmarks = []
        frames = []
        frame_count = 0
        max_frames = 300  # Limit processing to 100 frames

        # First read all frames and landmarks
        while cap.isOpened() and frame_count < max_frames:
        # while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"\rProcessing frame {frame_count}", end="")
                
            frames.append(frame)
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                all_landmarks.append(landmarks)
            else:
                all_landmarks.append(None)

        print(f"\nMediaPipe detection completed, processed {frame_count} frames")
        
        # Convert to numpy array for processing
        all_landmarks = np.array(all_landmarks)
        
        # Process landmarks based on different methods
        if method != 'original':
            print(f"Starting to apply {method} method to process landmarks...")
            processed_landmarks = self.apply_method(all_landmarks, method)
        else:
            processed_landmarks = all_landmarks

        # Draw processed results
        print("Starting to generate output video...")
        for i, frame in enumerate(frames):
            if processed_landmarks[i] is not None:
                # Convert processed landmarks to MediaPipe format
                landmark_list = landmark_pb2.NormalizedLandmarkList()
                for landmark_data in processed_landmarks[i]:
                    landmark = landmark_list.landmark.add()
                    landmark.x = float(landmark_data[0])
                    landmark.y = float(landmark_data[1])
                    landmark.z = float(landmark_data[2])
                    landmark.visibility = float(landmark_data[3])
                
                self.mp_drawing.draw_landmarks(
                    frame,
                    landmark_list,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            out.write(frame)

        cap.release()
        out.release()
        
        # ffmpeg_command = [
        #     "ffmpeg", "-y", "-i", temp_output_path,
        #     "-vcodec", "libx264", "-acodec", "aac", output_path
        # ]
        # subprocess.run(ffmpeg_command, check=True)
        print(f"Completed processing with {method} method\n")

    def apply_method(self, landmarks, method):
        if method == 'kalman':
            return self.apply_kalman_filter(landmarks)
        elif method == 'butterworth':
            return self.apply_butterworth_filter(landmarks)
        elif method == 'wiener':
            return self.apply_wiener_filter(landmarks)
        elif method == 'linear':
            return self.apply_linear_interpolation(landmarks)
        elif method == 'bilinear':
            return self.apply_bilinear_interpolation(landmarks)
        elif method == 'spline':
            return self.apply_spline_interpolation(landmarks)
        elif method == 'kriging':
            return self.apply_kriging_interpolation(landmarks)
        elif method == 'chebyshev':
            return self.apply_chebyshev_filter(landmarks)
        elif method == 'bessel':
            return self.apply_bessel_filter(landmarks)
        return landmarks

    def apply_kalman_filter(self, landmarks):
        print("Applying Kalman filter...")
        filtered_landmarks = np.copy(landmarks)
        
        # Apply Kalman filter to each keypoint
        for point_idx in range(landmarks.shape[1]):
            kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
            kf.F = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
            kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
            kf.R *= 0.1
            kf.Q *= 0.1
            
            # Get x,y coordinates for current keypoint across all frames
            points = landmarks[:, point_idx, :2]
            filtered_points = []
            
            for point in points:
                kf.predict()
                if point is not None and not np.any(np.isnan(point)):
                    kf.update(point)
                filtered_points.append(kf.x[:2].flatten())  # Ensure 1D array
            
            filtered_points = np.array(filtered_points)  # Convert to numpy array
            filtered_landmarks[:, point_idx, :2] = filtered_points
            filtered_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return filtered_landmarks

    def apply_butterworth_filter(self, landmarks):
        print("Applying Butterworth filter...")
        filtered_landmarks = np.copy(landmarks)
        
        # Design Butterworth filter
        order = 4  # Filter order
        cutoff = 0.1  # Cutoff frequency
        b, a = signal.butter(order, cutoff, 'low')
        
        # Apply filter to each keypoint
        for point_idx in range(landmarks.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = landmarks[:, point_idx, dim]
                # Apply filter
                filtered_signal = signal.filtfilt(b, a, signal_1d)
                filtered_landmarks[:, point_idx, dim] = filtered_signal
            filtered_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return filtered_landmarks

    def apply_wiener_filter(self, landmarks):
        print("Applying Wiener filter...")
        filtered_landmarks = np.copy(landmarks)
        
        # Apply Wiener filter to each keypoint
        for point_idx in range(landmarks.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = landmarks[:, point_idx, dim]
                # Apply Wiener filter
                filtered_signal = signal.wiener(signal_1d, mysize=5)
                filtered_landmarks[:, point_idx, dim] = filtered_signal
            filtered_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return filtered_landmarks

    def apply_linear_interpolation(self, landmarks):
        print("Applying linear interpolation...")
        interpolated_landmarks = np.copy(landmarks)
        
        # Interpolate each keypoint
        for point_idx in range(landmarks.shape[1]):
            # Get valid frame indices (non-None frames)
            valid_frames = np.where(landmarks[:, point_idx, 3] > 0.5)[0]
            if len(valid_frames) < 2:
                continue
                
            # Interpolate x,y,z coordinates separately
            for dim in range(3):
                f = interp1d(valid_frames, 
                           landmarks[valid_frames, point_idx, dim],
                           kind='linear',
                           fill_value="extrapolate")
                interpolated_landmarks[:, point_idx, dim] = f(np.arange(len(landmarks)))
            
            interpolated_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return interpolated_landmarks

    def apply_spline_interpolation(self, landmarks):
        print("Applying spline interpolation...")
        interpolated_landmarks = np.copy(landmarks)
        
        # Interpolate each keypoint
        for point_idx in range(landmarks.shape[1]):
            # Get valid frame indices
            valid_frames = np.where(landmarks[:, point_idx, 3] > 0.5)[0]
            if len(valid_frames) < 4:  # Spline interpolation needs at least 4 points
                continue
                
            # Interpolate x,y,z coordinates separately
            for dim in range(3):
                # Use cubic spline interpolation
                tck = splrep(valid_frames, 
                           landmarks[valid_frames, point_idx, dim],
                           k=3)
                interpolated_landmarks[:, point_idx, dim] = splev(np.arange(len(landmarks)), tck)
            
            interpolated_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return interpolated_landmarks

    def apply_bilinear_interpolation(self, landmarks):
        print("Applying bilinear interpolation...")
        interpolated_landmarks = np.copy(landmarks)
        
        # Interpolate each keypoint
        for point_idx in range(landmarks.shape[1]):
            # Get valid frame indices
            valid_frames = np.where(landmarks[:, point_idx, 3] > 0.5)[0]
            if len(valid_frames) < 2:
                continue
                
            # Create time grid
            time_grid = np.arange(len(landmarks))
            
            # Apply bilinear interpolation to x,y coordinates
            for dim in range(2):
                interpolated_landmarks[:, point_idx, dim] = griddata(
                    valid_frames, 
                    landmarks[valid_frames, point_idx, dim],
                    time_grid,
                    method='linear'
                )
            
            interpolated_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return interpolated_landmarks

    def apply_kriging_interpolation(self, landmarks):
        print("Applying Kriging interpolation...")
        interpolated_landmarks = np.copy(landmarks)
        
        # Interpolate each keypoint
        for point_idx in range(landmarks.shape[1]):
            # Get valid frame indices
            valid_frames = np.where(landmarks[:, point_idx, 3] > 0.5)[0]
            if len(valid_frames) < 3:  # Kriging needs at least 3 points
                continue
            
            # Apply Kriging interpolation to x,y coordinates
            for dim in range(2):
                try:
                    # Prepare data - ensure float64 type
                    x = valid_frames.astype(np.float64).reshape(-1, 1)  # Time as x coordinate
                    y = np.zeros_like(x)  # Virtual y coordinate
                    z = landmarks[valid_frames, point_idx, dim].astype(np.float64)  # Actual values
                    
                    # Create Kriging model
                    ok = OrdinaryKriging(
                        x.flatten(),
                        y.flatten(),
                        z,
                        variogram_model='gaussian',
                        verbose=False,
                        enable_plotting=False
                    )
                    
                    # Make predictions
                    grid_x = np.arange(len(landmarks), dtype=np.float64).reshape(-1, 1)
                    grid_y = np.zeros_like(grid_x)
                    z_pred, _ = ok.execute('points', grid_x.flatten(), grid_y.flatten())
                    
                    interpolated_landmarks[:, point_idx, dim] = z_pred
                except Exception as e:
                    print(f"Kriging interpolation failed, using spline interpolation instead: {e}")
                    if len(valid_frames) >= 4:
                        tck = splrep(valid_frames, landmarks[valid_frames, point_idx, dim], k=3)
                        interpolated_landmarks[:, point_idx, dim] = splev(np.arange(len(landmarks)), tck)
                    else:
                        f = interp1d(valid_frames, 
                                   landmarks[valid_frames, point_idx, dim],
                                   kind='linear',
                                   fill_value="extrapolate")
                        interpolated_landmarks[:, point_idx, dim] = f(np.arange(len(landmarks)))
            
            # Use spline interpolation for z coordinate
            if len(valid_frames) >= 4:
                tck = splrep(valid_frames, landmarks[valid_frames, point_idx, 2], k=3)
                interpolated_landmarks[:, point_idx, 2] = splev(np.arange(len(landmarks)), tck)
            else:
                f = interp1d(valid_frames, 
                           landmarks[valid_frames, point_idx, 2],
                           kind='linear',
                           fill_value="extrapolate")
                interpolated_landmarks[:, point_idx, 2] = f(np.arange(len(landmarks)))
            
            # Set visibility to 1 to indicate all points are visible
            interpolated_landmarks[:, point_idx, 3] = 1.0
        
        return interpolated_landmarks
    
    def apply_chebyshev_filter(self, landmarks):
        print("Applying Chebyshev filter...")
        filtered_landmarks = np.copy(landmarks)
        
        # Design Chebyshev filter
        order = 4  # Filter order
        ripple_db = 1.0  # Ripple in dB
        cutoff = 0.1  # Cutoff frequency
        b, a = signal.cheby1(order, ripple_db, cutoff, 'low')
        
        # Apply filter to each keypoint
        for point_idx in range(landmarks.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = landmarks[:, point_idx, dim]
                # Apply filter
                filtered_signal = signal.filtfilt(b, a, signal_1d)
                filtered_landmarks[:, point_idx, dim] = filtered_signal
            filtered_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return filtered_landmarks

    def apply_bessel_filter(self, landmarks):
        print("Applying Bessel filter...")
        filtered_landmarks = np.copy(landmarks)
        
        # Design Bessel filter
        order = 4  # Filter order
        cutoff = 0.1  # Cutoff frequency
        b, a = signal.bessel(order, cutoff, 'low')
        
        # Apply filter to each keypoint
        for point_idx in range(landmarks.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = landmarks[:, point_idx, dim]
                # Apply filter
                filtered_signal = signal.filtfilt(b, a, signal_1d)
                filtered_landmarks[:, point_idx, dim] = filtered_signal
            filtered_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return filtered_landmarks


def main():
    processor = PoseProcessor()
    input_video = ""
    
    methods = ['original', 'kalman', 'butterworth', 'wiener', 
              'linear', 'bilinear', 'spline', 'kriging','chebyshev', 'bessel']
    
    for method in methods:
        processor.process_video(input_video, method)

if __name__ == "__main__":
    main()