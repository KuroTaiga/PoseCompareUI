import os
import torch
import numpy as np
from tqdm import tqdm

def build_frequency_matrix(poses, eps=5.0, normalized=False):
    """
    Compute the frequency matrix based on changes in Euclidean distance between joints.
    
    Args:
        poses: numpy array of shape (T, N, 2) with (x, y) coordinates.
        eps: Threshold for significant change (pixels or normalized units).
        normalized: Whether coordinates are normalized.
    
    Returns:
        freq_matrix: (N, N) matrix with the number of significant changes.
    """
    T, N, _ = poses.shape
    freq_matrix = np.zeros((N, N), dtype=np.float32)
    
    if normalized:
        eps = eps / 1000.0  # Adjust for normalized coordinates
    
    for i in range(N):
        for j in range(i + 1, N):  # Symmetric matrix
            distances = np.linalg.norm(poses[:, i, :] - poses[:, j, :], axis=1)
            changes = np.sum(np.abs(distances[1:] - distances[:-1]) > eps)
            freq_matrix[i, j] = changes
            freq_matrix[j, i] = changes
    
    return freq_matrix

def process_all_videos_for_frequency(input_dir, output_dir, eps=5.0, normalized=False):
    """
    Process all .pt files in input_dir, compute frequency matrices, and save to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_files = [os.path.join(root, f) for root, _, files in os.walk(input_dir)
                 for f in files if f.endswith('.pt')]
    
    for file_path in tqdm(all_files, desc="Processing videos for frequency matrix"):
        try:
            tensor = torch.load(file_path, weights_only=False)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            poses = tensor.numpy()[..., :2]  # (T, 17, 2)
            freq_matrix = build_frequency_matrix(poses, eps=eps, normalized=normalized)
            
            # Save with mirrored directory structure
            rel_path = os.path.relpath(file_path, input_dir)
            output_file = os.path.join(output_dir, rel_path).replace('.pt', '_freq.pt')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            torch.save(torch.from_numpy(freq_matrix), output_file)
            print(f"Saved frequency matrix: {output_file}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    input_dir = "/home/bizon/zihan/Blend_Rendered_Landmarks"
    output_dir = "/home/bizon/zihan/Frequency_Matrices"
    eps = 5.0  # For pixel coordinates
    normalized = False  # Set to True for normalized tensors
    process_all_videos_for_frequency(input_dir, output_dir, eps=eps, normalized=normalized)

if __name__ == "__main__":
    main()