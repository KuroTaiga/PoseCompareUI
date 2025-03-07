import cv2
import os
import sys
import torch
from multiprocessing import cpu_count, Pool, Process
from typing import List, Optional, Sequence, Union
import torch.nn as nn
import glob
import torch.nn.functional as F
import tqdm
import numpy as np
import json
from worker_pool import WorkerPool
from sapiens_classes_and_consts import COCO_KPTS_COLORS, COCO_SKELETON_INFO
from sapiens_util import udp_decode
class AdhocImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, shape=None, mean=None, std=None):
        self.image_list = image_list
        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape = shape
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None

    def __len__(self):
        return len(self.image_list)
    
    def _preprocess(self, img):
        if self.shape:
            img = cv2.resize(img, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        if self.mean is not None and self.std is not None:
            mean=self.mean.view(-1, 1, 1)
            std=self.std.view(-1, 1, 1)
            img = (img - mean) / std
        return img
    
    def __getitem__(self, idx):
        orig_img_dir = self.image_list[idx]
        orig_img = cv2.imread(orig_img_dir)
        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img = self._preprocess(orig_img)
        return orig_img_dir, orig_img, img

class SapiensProcessor():
    def __init__(self,checkpoint,device = "cuda:0",batch_size = 8,heatmap_scale = 4,shape = [1024,768],output_folder = "sapiens_output"):
        self.estimator = torch.jit.load(checkpoint)
        self.dtype = torch.float32  # TorchScript models use float32
        self.estimator = self.estimator.to(device)
        self.batch_size = batch_size
        self.heatmap_scale = heatmap_scale
        self.shape = shape
        self.output_root = output_folder
        pass

    def process_video(self, input_path,output_path,method = "original", output_format="mp4",kpt_thr = 0.3,radius = 6,thickness = 3):
        scale = self.heatmap_scale
        
        input_shape = (3,)+tuple(self.shape)
        image_names = []
        imgs_path = self.vid_to_img(input_path)
        out_img_folder = os.path.join(self.output_root, os.path.basename(imgs_path))
        if os.path.isdir(imgs_path):
            input_dir = imgs_path
            image_names = [
                img_name for img_name in os.listdir(input_dir) if img_name.endswith(".jpg") or img_name.endswith(".png")
            ]
        else:
            print(f"Error: iamges file path '{imgs_path}' does not exist.")
            return
        inference_dataset = AdhocImageDataset(
        [os.path.join(input_dir, img_name) for img_name in image_names],
        )  # do not provide preprocess args for detector as we use mmdet
        inference_dataloader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(min(self.batch_size, cpu_count()) // 4, 4),
        )
        pose_preprocess_pool = WorkerPool(
            self.preprocess_pose, processes=max(min(self.batch_size, cpu_count()), 1)
        )
        img_save_pool = WorkerPool(
            self.img_save_and_vis, processes=max(min(self.batch_size, cpu_count()), 1)
        )
        KPTS_COLORS = COCO_KPTS_COLORS
        SKELETON_INFO = COCO_SKELETON_INFO
        for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
            enumerate(inference_dataloader), total=len(inference_dataloader)):
            orig_img_shape = batch_orig_imgs.shape
            valid_images_len = len(batch_orig_imgs)
            bboxes_batch = [[] for _ in range(len(batch_orig_imgs))]
            assert len(bboxes_batch) == valid_images_len
            for i, bboxes in enumerate(bboxes_batch):
                if len(bboxes) == 0:
                    bboxes_batch[i] = np.array(
                        [[0, 0, orig_img_shape[1], orig_img_shape[2]]] # orig_img_shape: B H W C
                    )
            img_bbox_map = {}
            for i, bboxes in enumerate(bboxes_batch):
                img_bbox_map[i] = len(bboxes)

            args_list = [
                (
                    i,
                    bbox_list,
                    (input_shape[1], input_shape[2]),
                    [123.5, 116.5, 103.5],
                    [58.5, 57.0, 57.5],
                )
                for i, bbox_list in zip(batch_orig_imgs.numpy(), bboxes_batch)
            ]
            pose_ops = pose_preprocess_pool.run(args_list)
            pose_imgs, pose_img_centers, pose_img_scales = [], [], []
            for op in pose_ops:
                pose_imgs.extend(op[0])
                pose_img_centers.extend(op[1])
                pose_img_scales.extend(op[2])

            n_pose_batches = (len(pose_imgs) + self.batch_size - 1) // self.batch_size

            # use this to tell torch compiler the start of model invocation as in 'flip' mode the tensor output is overwritten
            torch.compiler.cudagraph_mark_step_begin()  
            pose_results = []
            for i in range(n_pose_batches):
                imgs = torch.stack(
                    pose_imgs[i * self.batch_size : (i + 1) * self.batch_size], dim=0
                )
                valid_len = len(imgs)
                imgs = F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, self.batch_size - imgs.shape[0]), value=0)
                pose_results.extend(
                    self.batch_inference_topdown(self.estimator, imgs, dtype=self.dtype)[:valid_len]
                )

            batched_results = []
            for _, bbox_len in img_bbox_map.items():
                result = {
                    "heatmaps": pose_results[:bbox_len].copy(),
                    "centres": pose_img_centers[:bbox_len].copy(),
                    "scales": pose_img_scales[:bbox_len].copy(),
                }
                batched_results.append(result)
                del (
                    pose_results[:bbox_len],
                    pose_img_centers[:bbox_len],
                    pose_img_scales[:bbox_len],
                )

            assert len(batched_results) == len(batch_orig_imgs)
            args_list = [
                (
                    i.numpy(),
                    r,
                    os.path.join(out_img_folder, os.path.basename(img_name)),
                    (input_shape[2], input_shape[1]),
                    scale,
                    KPTS_COLORS,
                    kpt_thr,
                    radius,
                    SKELETON_INFO,
                    thickness,
                )
                for i, r, img_name in zip(
                    batch_orig_imgs[:valid_images_len],
                    batched_results[:valid_images_len],
                    batch_image_name,
                )
            ]
            img_save_pool.run_async(args_list)
        pose_preprocess_pool.finish()
        img_save_pool.finish()
        print("writting video")
        self.img_to_vid(out_img_folder,output_path)

    def batch_inference_topdown(self,model: nn.Module,imgs: List[Union[np.ndarray, str]],dtype=torch.bfloat16,flip=False,):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            heatmaps = model(imgs.cuda())
            if flip:
                heatmaps_ = model(imgs.to(dtype).cuda().flip(-1))
                heatmaps = (heatmaps + heatmaps_) * 0.5
            imgs.cpu()
        return heatmaps.cpu()

    def img_save_and_vis(
        img, results, output_path, input_shape, heatmap_scale, kpt_colors, kpt_thr, radius, skeleton_info, thickness
    ):
        # pred_instances_list = split_instances(result)
        heatmap = results["heatmaps"]
        centres = results["centres"]
        scales = results["scales"]
        img_shape = img.shape
        instance_keypoints = []
        instance_scores = []
        # print(scales[0], centres[0])
        for i in range(len(heatmap)):
            result = udp_decode(
                heatmap[i].cpu().unsqueeze(0).float().data[0].numpy(),
                input_shape,
                (int(input_shape[0] / heatmap_scale), int(input_shape[1] / heatmap_scale)),
            )

            keypoints, keypoint_scores = result
            keypoints = (keypoints / input_shape) * scales[i] + centres[i] - 0.5 * scales[i]
            instance_keypoints.append(keypoints[0])
            instance_scores.append(keypoint_scores[0])

        pred_save_path = output_path.replace(".jpg", ".json").replace(".png", ".json")

        with open(pred_save_path, "w") as f:
            json.dump(
                dict(
                    instance_info=[
                        {
                            "keypoints": keypoints.tolist(),
                            "keypoint_scores": keypoint_scores.tolist(),
                        }
                        for keypoints, keypoint_scores in zip(
                            instance_keypoints, instance_scores
                        )
                    ]
                ),
                f,
                indent="\t",
            )
        # img = pyvips.Image.new_from_array(img)
        instance_keypoints = np.array(instance_keypoints).astype(np.float32)
        instance_scores = np.array(instance_scores).astype(np.float32)

        keypoints_visible = np.ones(instance_keypoints.shape[:-1])
        for kpts, score, visible in zip(
            instance_keypoints, instance_scores, keypoints_visible
        ):
            kpts = np.array(kpts, copy=False)

            if (
                kpt_colors is None
                or isinstance(kpt_colors, str)
                or len(kpt_colors) != len(kpts)
            ):
                raise ValueError(
                    f"the length of kpt_color "
                    f"({len(kpt_colors)}) does not matches "
                    f"that of keypoints ({len(kpts)})"
                )

            # draw each point on image
            for kid, kpt in enumerate(kpts):
                if score[kid] < kpt_thr or not visible[kid] or kpt_colors[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = kpt_colors[kid]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color[::-1])
                img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), color, -1)
            
            # draw skeleton
            for skid, link_info in skeleton_info.items():
                pt1_idx, pt2_idx = link_info['link']
                color = link_info['color'][::-1] # BGR

                pt1 = kpts[pt1_idx]; pt1_score = score[pt1_idx]
                pt2 = kpts[pt2_idx]; pt2_score = score[pt2_idx]

                if pt1_score > kpt_thr and pt2_score > kpt_thr:
                    x1_coord = int(pt1[0]); y1_coord = int(pt1[1])
                    x2_coord = int(pt2[0]); y2_coord = int(pt2[1])
                    cv2.line(img, (x1_coord, y1_coord), (x2_coord, y2_coord), color, thickness=thickness)

        cv2.imwrite(output_path, img)

    def preprocess_pose(self,orig_img, bboxes_list, input_shape, mean, std):
        """Preprocess pose images and bboxes."""
        preprocessed_images = []
        centers = []
        scales = []
        for bbox in bboxes_list:
            img, center, scale = self.top_down_affine_transform(orig_img.copy(), bbox)
            img = cv2.resize(
                img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)
            img = torch.from_numpy(img)
            img = img[[2, 1, 0], ...].float()
            mean = torch.Tensor(mean).view(-1, 1, 1)
            std = torch.Tensor(std).view(-1, 1, 1)
            img = (img - mean) / std
            preprocessed_images.append(img)
            centers.extend(center)
            scales.extend(scale)
        return preprocessed_images, centers, scales
    def vid_to_img(video_path):
        if not os.path.exists(video_path):
            print(f"Error: The video file '{video_path}' does not exist.")
            return
        
        # Create output directory
        output_dir = os.path.splitext(video_path)[0]  # Remove extension
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Stop if no more frames
            
            # Construct file name
            filename = os.path.join(output_dir, f"{frame_number:06d}.jpg")
            
            # Save frame as JPEG
            cv2.imwrite(filename, frame)
            frame_number += 1
        
        # Release the video capture object
        cap.release()
        return output_dir
    def img_to_vid(img_folder, output_path,fps=30):
        images = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))
        if not images:
            print(f"Error: No images found in '{img_folder}'.")
            return
        frame = cv2.imread(images[0])
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for image_path in images:
            frame = cv2.imread(image_path)
            out.write(frame)
        out.release()
        print(f"Video saved to {output_path}")


