import os
import torch
import numpy as np
from tqdm import tqdm

def build_relation_matrix(poses, eps=5.0, normalized=False):
    """
    Compute the relation matrix for each frame based on spatial relationships between joints.
    
    Args:
        poses: numpy array of shape (T, N, 2) with (x, y) coordinates.
        eps: Threshold to determine 'inline' status (pixels or normalized units).
        normalized: Whether coordinates are normalized (0 to 1).
    
    Returns:
        relation_tensor: (T, N, N) tensor with relation codes (0-8).
    """
    T, N, _ = poses.shape
    relation_tensor = np.zeros((T, N, N), dtype=np.int8)
    
    # Adjust eps for normalized coordinates (assuming pixel range ~1000)
    if normalized:
        eps = eps / 1000.0
    
    for t in range(T):
        for i in range(N):
            x_i, y_i = poses[t, i, 0], poses[t, i, 1]
            for j in range(N):
                if i == j:
                    relation_tensor[t, i, j] = 4  # Inline-inline (self-relation)
                    continue
                x_j, y_j = poses[t, j, 0], poses[t, j, 1]
                # Vertical: 0 (above), 1 (inline), 2 (below)
                V = 0 if (y_i - y_j) > eps else (2 if (y_j - y_i) > eps else 1)
                # Horizontal: 0 (right), 1 (inline), 2 (left)
                H = 0 if (x_j - x_i) > eps else (2 if (x_i - x_j) > eps else 1)
                relation_tensor[t, i, j] = 3 * V + H  # Codes 0-8
    return relation_tensor

def process_all_videos_for_relation(input_dir, output_dir, eps=5.0, normalized=False):
    """
    Process all .pt files in input_dir, compute relation matrices, and save to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_files = [os.path.join(root, f) for root, _, files in os.walk(input_dir)
                 for f in files if f.endswith('.pt')]
    
    for file_path in tqdm(all_files, desc="Processing videos for relation matrix"):
        try:
            tensor = torch.load(file_path, weights_only=False)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            poses = tensor.numpy()[..., :2]  # (T, 17, 2)
            relation_tensor = build_relation_matrix(poses, eps=eps, normalized=normalized)
            
            # Save with mirrored directory structure
            rel_path = os.path.relpath(file_path, input_dir)
            output_file = os.path.join(output_dir, rel_path).replace('.pt', '_relation.pt')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            torch.save(torch.from_numpy(relation_tensor), output_file)
            print(f"Saved relation matrix: {output_file}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    input_dir = "/home/bizon/zihan/Blend_Rendered_Landmarks"
    output_dir = "/home/bizon/zihan/Relation_Matrices"
    eps = 0  # For pixel coordinates; adjust as needed
    normalized = False  # Set to True for normalized tensors
    process_all_videos_for_relation(input_dir, output_dir, eps=eps, normalized=normalized)

if __name__ == "__main__":
    main()