import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Joint names and order (0 to 16)
joint_names = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]
original_order = list(range(17))

def visualize_frequency_matrix(freq_file_path, poses_file_path, output_dir):
    """
    Generate an interactive heatmap for the frequency matrix with colored separators.
    """
    try:
        freq_matrix = torch.load(freq_file_path).numpy()  # (17, 17)
        poses = torch.load(poses_file_path).numpy()[..., :2]  # (T, 17, 2)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        heatmap = ax.imshow(freq_matrix, cmap='hot_r', aspect='equal')
        plt.colorbar(heatmap, ax=ax, label='Movement Frequency (Changes)')
        
        # Add colored lines to separate sections
        sections = [
            (0, 5, 'purple'),  # Head: 0-4
            (5, 11, 'red'),    # Arms: 5-10
            (11, 17, 'green')  # Legs: 11-16
        ]
        for start, end, color in sections:
            ax.hlines(y=start - 0.5, xmin=-0.5, xmax=16.5, color=color, linewidth=2)
            ax.hlines(y=end - 0.5, xmin=-0.5, xmax=16.5, color=color, linewidth=2)
            ax.vlines(x=start - 0.5, ymin=-0.5, ymax=16.5, color=color, linewidth=2)
            ax.vlines(x=end - 0.5, ymin=-0.5, ymax=16.5, color=color, linewidth=2)
        
        ax.set_xticks(np.arange(17))
        ax.set_yticks(np.arange(17))
        ax.set_xticklabels(original_order, rotation=90, fontsize=8)
        ax.set_yticklabels(original_order, fontsize=8)
        ax.set_title(f"Movement Frequency Heatmap\n{os.path.basename(freq_file_path)}")
        
        # Store poses for interactivity
        fig.poses = poses
        
        def on_click(event):
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                i, j = int(round(event.ydata)), int(round(event.xdata))
                if 0 <= i < 17 and 0 <= j < 17:
                    distances = np.linalg.norm(poses[:, i, :] - poses[:, j, :], axis=1)
                    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
                    ax_dist.plot(distances)
                    ax_dist.set_title(f"Distance: {joint_names[i]} to {joint_names[j]}")
                    ax_dist.set_xlabel("Frame")
                    ax_dist.set_ylabel("Distance")
                    plt.show()
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Save PNG and interactive figure
        rel_path = os.path.relpath(freq_file_path, output_dir)
        output_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(output_subdir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(freq_file_path))[0]
        output_png = os.path.join(output_subdir, f"{base_name}_heatmap.png")
        plt.tight_layout()
        plt.savefig(output_png, dpi=150)
        output_pickle = os.path.join(output_subdir, f"{base_name}_heatmap.fig.pkl")
        with open(output_pickle, 'wb') as f:
            pickle.dump(fig, f)
        plt.close(fig)
        
        print(f"Saved frequency heatmap PNG: {output_png}")
        print(f"Saved interactive figure: {output_pickle}")
        return output_png
    except Exception as e:
        print(f"Error visualizing {freq_file_path}: {e}")
        return None

def main():
    freq_dir = "/home/bizon/zihan/Frequency_Matrices"
    poses_dir = "/home/bizon/zihan/Blend_Rendered_Landmarks"
    output_dir = "/home/bizon/zihan/Video_Heatmaps_Frequency314"
    
    all_freq_files = [os.path.join(root, f) for root, _, files in os.walk(freq_dir)
                      for f in files if f.endswith('_freq.pt')]
    all_poses_files = [os.path.join(poses_dir, os.path.relpath(f, freq_dir).replace('_freq.pt', '.pt'))
                       for f in all_freq_files]
    
    for freq_file, poses_file in zip(all_freq_files, all_poses_files):
        visualize_frequency_matrix(freq_file, poses_file, output_dir)

if __name__ == "__main__":
    main()