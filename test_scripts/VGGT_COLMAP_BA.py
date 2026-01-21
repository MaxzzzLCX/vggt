# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings - REMOVED CUDA-specific backend settings
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap
from PIL import Image


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--mask", action="store_true", default=False, help="Whether to use masks")
    parser.add_argument("--mask_dir", type=str, default="masks", help="Directory containing the mask images")
    parser.add_argument("--scale_pointcloud", action="store_true", default=False, help="Scale point cloud to real-world units")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # Set use_ba default to False as it requires extra dependencies and is slow on CPU
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=False, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=2.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def load_masks(mask_dir, image_path_list, device, resolution):
    """Loads and resizes masks corresponding to a list of images."""

    if mask_dir is None:
        raise ValueError("mask_dir is not provided, but load_masks is called.")
    
    print(f"Loading and resizing masks from {mask_dir}...")
    mask_tensors = []
    for img_path in image_path_list:
        # Assumes mask has the same name but with .png extension
        mask_name = os.path.splitext(os.path.basename(img_path))[0] + "_mask_1.png"
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Could not find mask {mask_path} for image {img_path}")
            
        mask_pil = Image.open(mask_path).convert('L') # Load as grayscale PIL image
        
        # Write the raw image as np array before resizing for debugging in txt format
        mask_np = np.array(mask_pil)
        raw_output_path = os.path.join("debug_masks", f"raw_{mask_name[:-4]}_{mask_np.shape[0]}x{mask_np.shape[1]}.txt")
        os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
        np.savetxt(raw_output_path, mask_np, fmt='%d')

        # --- FIX: Resize the mask to a consistent resolution ---
        mask_pil = mask_pil.resize((resolution, resolution), Image.NEAREST)
        mask_np = np.array(mask_pil)
        masked_output_path = os.path.join("debug_masks", f"masked_{mask_name[:-4]}_{mask_np.shape[0]}x{mask_np.shape[1]}.txt")
        os.makedirs(os.path.dirname(masked_output_path), exist_ok=True)
        np.savetxt(masked_output_path, mask_np, fmt='%d')

        mask_np = np.array(mask_pil) > 128 # Convert to boolean (True for white, False for black)
        mask_tensors.append(torch.from_numpy(mask_np))
        
    masks = torch.stack(mask_tensors).to(device)
    print(f"Loaded and resized {len(masks)} masks to {resolution}x{resolution}.")
    print(f"Shape of masks tensor is {masks.shape}")
    return masks

def run_VGGT(model, images, dtype, device, resolution=518):
    # images: [S, 3, H, W]
    # model: VGGT model instance
    # device: 'cpu' or 'cuda'

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.amp.autocast(device_type=device, dtype=dtype):
                # The model aggregator expects a batch dimension B, where B=1
                images_b = images[None]  # add batch dimension -> (1, S, 3, H, W)
                aggregated_tokens_list, ps_idx = model.aggregator(images_b)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_b, ps_idx)
        else:
            print("Running on CPU...")
            # Run on CPU with default precision (float32)
            images_b = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_b)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_b, ps_idx)

    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf

def load_gt_depthmap(image_path_list):
    """
    Load ground truth depth maps of each image from a folder
    Each depth map is stored in a .npy file with the same base name as the image.
    """
    depth_maps = []
    for img_path in image_path_list:
        filename = os.path.basename(img_path)
        depth_filename = f"{filename[8:-7]}depth.npy"  # Change resized_XXX_img.png to XXX_depth.npy

        # Depth_path is in the parent's parent directory named "images"
        depth_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), "images", depth_filename)

        print(depth_path)
        depth_maps.append(np.load(depth_path))

    # print(f"Depth map shape: {len(depth_maps)} {depth_maps[0].shape}")

    depth_maps = np.stack(depth_maps, axis=0)  # Shape: (num_images, H, W)
    print(f"After stacking, has shape {depth_maps.shape}")
    return depth_maps

def calculate_conversion_factor(depth_map, gt_depth_map, combined_mask):
    """Calculate the conversion factor between predicted depth and ground truth depth."""
    
    # Combine mask (combination of sam and depth_conf map from VGGT) and depth validity mask (if depth=0 in GT, it is background, not object)
    mask = (combined_mask > 0) & (gt_depth_map > 0)

    # Apply the SAM mask to both depth maps
    depth_map_masked = depth_map[mask]
    gt_depth_map_masked = gt_depth_map[mask]

    # Randomly sample 100 points and compute the ratio, for testing purpose
    if len(depth_map_masked) > 100:
        sample_indices = np.random.choice(len(depth_map_masked), size=100, replace=False)
        depth_map_masked = depth_map_masked[sample_indices]
        gt_depth_map_masked = gt_depth_map_masked[sample_indices]
        
        print(f"Predicted):", depth_map_masked)
        print(f"GT:", gt_depth_map_masked)
        print(f"Ratio:", gt_depth_map_masked / depth_map_masked)

    # Compute the conversion factor
    conversion_factor = np.mean(gt_depth_map_masked / depth_map_masked)
    var = np.var(gt_depth_map_masked / depth_map_masked)

    print(f"Conversion factor: {conversion_factor}, Variance: {var}")

    return conversion_factor

def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    # dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    # model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    # model.eval()
    # model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "resized_images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in by its larger dimension of image's width and height, while running VGGT with 518
    vggt_fixed_resolution = 518
    # larger_dimension = max(img_load_resolution, vggt_fixed_resolution)
    img_load_resolution = vggt_fixed_resolution
    print(f"Loading images at resolution: {vggt_fixed_resolution}")

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, device, vggt_fixed_resolution)
    print(f"VGGT depth_map shape: {depth_map.shape}")

    # --- FIX: Convert extrinsic from (N, 3, 4) to (N, 4, 4) ---
    # VGGT outputs camera matrices as [R|t] in (3, 4) format
    # COLMAP expects homogeneous (4, 4) format with bottom row [0, 0, 0, 1]
    if extrinsic.shape[1:] == (3, 4):
        num_frames = extrinsic.shape[0]
        extrinsic_4x4 = np.zeros((num_frames, 4, 4), dtype=extrinsic.dtype)
        extrinsic_4x4[:, :3, :] = extrinsic  # Copy [R|t]
        extrinsic_4x4[:, 3, 3] = 1.0  # Set bottom-right to 1
        extrinsic = extrinsic_4x4
        print(f"Converted extrinsic to homogeneous format: {extrinsic.shape}")

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.use_ba:
        """
        NOTE: this section is still experimental. There bugs still remain, specifically the handling of masks.
        Do not use it.
        """

        if device == "cpu":
            print("Warning: Bundle Adjustment (BA) is very slow on CPU.")
        
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        # --- NEW: Load SAM masks if requested ---
        ba_masks = None
        if args.mask:
            print(f"Loading masks from {args.mask_dir} for Bundle Adjustment...")
            # Load masks at the same resolution as the images being passed to the tracker
            segment_masks = load_masks(args.mask_dir, image_path_list, device, img_load_resolution)
            
            # The tracker expects masks at the model's resolution (518x518)
            ba_masks = F.interpolate(
                segment_masks[:, None].float(), size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="nearest"
            ).squeeze(1) # Result is a (S, H, W) float tensor (0.0 or 1.0)
            print("Applied SAM2 masks to the tracker.")

        # Note: predict_tracks may have its own internal CUDA dependencies.
        # This change makes the script runnable, but predict_tracks might still fail on CPU.

        # Predicting Tracks
        # Using VGGSfM tracker instead of VGGT tracker for efficiency
        # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
        # Will be fixed in VGGT v2

        # You can also change the pred_tracks to tracks from any other methods
        # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
        pred_tracks, pred_vis_scores, pred_confs, points_3d_tracked, points_rgb = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            keypoint_extractor="aliked+sp",
            fine_tracking=args.fine_tracking,
        )

        # DEBUG: Check raw shapes
        print(f"Raw pred_tracks shape: {pred_tracks.shape}")  # (S, N, 2)
        print(f"Raw points_3d_tracked shape: {points_3d_tracked.shape if points_3d_tracked is not None else None}")
        
        # --- No need to add depth dimension! ---
        # The function expects tracks to be (num_frames, num_tracks, 2) with just [x, y]
        # The 3D information comes from points_3d_tracked argument
        
        num_frames, num_tracks = pred_tracks.shape[:2]
        
        # Verify tracks are 2D only
        if pred_tracks.shape[-1] != 2:
            raise ValueError(f"Expected tracks with shape (N, P, 2), got {pred_tracks.shape}")
        
        print(f"Track shape: {pred_tracks.shape}")  # Should be (N_frames, N_tracks, 2)
        print(f"Visibility shape: {pred_vis_scores.shape}")  # Should be (N_frames, N_tracks)
        print(f"Extrinsic shape: {extrinsic.shape}")  # Should be (N_frames, 4, 4)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # rescale the intrinsic matrix
        intrinsic[:, :2, :] *= scale
        
        # --- Create track_mask from visibility ---
        track_mask = pred_vis_scores > args.vis_thresh
        print(f"Initial valid track-points based on visibility threshold ({args.vis_thresh}): {np.sum(track_mask)}")

        # --- Apply SAM masks if provided ---
        if args.mask:
            num_frames_from_masks = segment_masks.shape[0]
            num_frames_to_process = min(num_frames, num_frames_from_masks)
            
            if num_frames > num_frames_from_masks:
                print(f"Warning: Tracker returned {num_frames} frames, "
                      f"but only {num_frames_from_masks} masks available.")
            
            # Sample mask at track coordinates
            # pred_tracks is (num_frames, num_tracks, 2)
            track_coords_xy = pred_tracks[:num_frames_to_process, :, :]  # Already 2D
            sam_track_mask = np.zeros((num_frames_to_process, num_tracks), dtype=bool)

            for i in range(num_frames_to_process):
                coords_x = np.round(track_coords_xy[i, :, 0]).astype(np.int32)
                coords_y = np.round(track_coords_xy[i, :, 1]).astype(np.int32)
                coords_x = np.clip(coords_x, 0, vggt_fixed_resolution - 1)
                coords_y = np.clip(coords_y, 0, vggt_fixed_resolution - 1)
                sam_track_mask[i, :] = segment_masks[i, coords_y, coords_x].cpu().numpy()

            print(f"SAM mask coverage: {sam_track_mask.sum()} / {sam_track_mask.size} points inside mask")

            # Pad to full frame count if needed
            if num_frames_to_process < num_frames:
                padded_sam_mask = np.zeros((num_frames, num_tracks), dtype=bool)
                padded_sam_mask[:num_frames_to_process, :] = sam_track_mask
                sam_track_mask = padded_sam_mask

            # Combine visibility and SAM masks
            final_track_mask = np.logical_and(track_mask, sam_track_mask)
            
            # Remove tracks with < 3 valid observations
            valid_count_per_track = final_track_mask.sum(axis=0)  # Sum over frames
            tracks_with_sufficient_obs = valid_count_per_track >= 3
            final_track_mask[:, ~tracks_with_sufficient_obs] = False
            
            print(f"Valid track-points after SAM masking: {final_track_mask.sum()}")
            print(f"Tracks with ≥3 observations: {tracks_with_sufficient_obs.sum()} / {num_tracks}")
        else:
            final_track_mask = track_mask

        # Verify dimensions
        print(f"\nPassing to COLMAP:")
        print(f"  points_3d_tracked: {points_3d_tracked.shape}")  # (P, 3)
        print(f"  extrinsic: {extrinsic.shape}")  # (N, 4, 4)
        print(f"  intrinsic: {intrinsic.shape}")  # (N, 3, 3)
        print(f"  pred_tracks: {pred_tracks.shape}")  # (N, P, 2)
        print(f"  final_track_mask: {final_track_mask.shape}")  # (N, P)
        print(f"  image_size: {image_size}")  # (2,)

        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d_tracked,  # (num_tracks, 3) - 3D positions
            extrinsic,          # (num_frames, 4, 4)
            intrinsic,          # (num_frames, 3, 3)
            pred_tracks,        # (num_frames, num_tracks, 2) - 2D pixel coords only!
            image_size,
            masks=final_track_mask,  # (num_frames, num_tracks)
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
        
        # --- FIX: Extract refined 3D points and colors from reconstruction ---
        points_3d_ba = []
        points_rgb_ba = []
        
        for point3D_id, point3D in reconstruction.points3D.items():
            points_3d_ba.append(point3D.xyz)
            points_rgb_ba.append(point3D.color)
        
        points_3d = np.array(points_3d_ba)
        points_rgb = np.array(points_rgb_ba)
        
        print(f"Extracted {len(points_3d)} refined 3D points from BA reconstruction")
        
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        ## Masking (Mask is generated by SAM2)
        # (1) Mask-out the low-confidence depth values
        conf_mask = depth_conf >= conf_thres_value
        print(f"Applying Confidence Mask with threshold {conf_thres_value}")

        # (2) Combine the confidence mask with SAM2 mask, if provided
        if args.mask:
            print(f"Loading masks from {args.mask_dir}...")
            # --- FIX: Pass the target resolution to load_masks ---
            segment_masks = load_masks(args.mask_dir, image_path_list, device, img_load_resolution) # (num_images, H, W) boolean tensor
            
            # The masks are now already at 1024x1024, so we still need to resize to the model's resolution
            segment_masks_resized = F.interpolate(
                segment_masks[:, None].float(), size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="nearest"
            ).squeeze(1).bool() # Result is (S, H, W) tensor
            
            segment_masks_np = segment_masks_resized.cpu().numpy()
            
            # Now both operands are NumPy arrays
            combined_mask = np.logical_and(conf_mask, segment_masks_np)
            print("Applied SAM2 masks") 
        else:
            combined_mask = conf_mask

        # at most writing 100000 3d points to colmap reconstruction object
        combined_mask = randomly_limit_trues(combined_mask, max_points_for_colmap)

        points_3d = points_3d[combined_mask]
        points_xyf = points_xyf[combined_mask]
        points_rgb = points_rgb[combined_mask]

        
        if args.scale_pointcloud:
            print("Scaling point cloud to real-world units using GT depth maps...")

            # Account for physical scale (VGGT predicts depth up to scale)
            # depth_map is of shape (# images, H, W, 1)
            # Load GT depth maps with the same dimensions

            gt_depth_maps = load_gt_depthmap(image_path_list)
            conversion_factor = calculate_conversion_factor(depth_map, gt_depth_maps, combined_mask)

            # Scale the entire point cloud by the conversion factor
            points_3d *= conversion_factor

            # Also have to scale the camera translation accordingly
            extrinsic_scaled = extrinsic.copy()
            extrinsic_scaled[:, :3, 3] *= conversion_factor
            extrinsic = extrinsic_scaled


        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    folder_name = f"sparse"

    if args.mask:
        folder_name += "_sam"
    else:
        folder_name += "_nosam"

    if args.scale_pointcloud:
        folder_name += "_scaled"
    else:
        folder_name += "_unscaled"

    if args.use_ba:
        folder_name += "_ba"
    
    folder_name += f"_conf{args.conf_thres_value}"

    sparse_reconstruction_dir = os.path.join(args.scene_dir, folder_name)

    # if args.mask:
    #     if args.use_ba:
    #         print(f"Saving reconstruction to {args.scene_dir}/sparse_sam_ba_conf{args.conf_thres_value}")
    #         sparse_reconstruction_dir = os.path.join(args.scene_dir, f"sparse_sam_ba_conf{args.conf_thres_value}")
    #     else:
    #         print(f"Saving reconstruction to {args.scene_dir}/sparse_sam_conf{args.conf_thres_value}")
    #         sparse_reconstruction_dir = os.path.join(args.scene_dir, f"sparse_sam_conf{args.conf_thres_value}")
    # else:
    #     if args.use_ba:
    #         print(f"Saving reconstruction to {args.scene_dir}/sparse_nosam_ba_conf{args.conf_thres_value}")
    #         sparse_reconstruction_dir = os.path.join(args.scene_dir, f"sparse_nosam_ba_conf{args.conf_thres_value}")
    #     else:
    #         print(f"Saving reconstruction to {args.scene_dir}/sparse_nosam_conf{args.conf_thres_value}")
    #         sparse_reconstruction_dir = os.path.join(args.scene_dir, f"sparse_nosam_conf{args.conf_thres_value}")
    
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    print(f"Exporting {len(points_3d)} points to {os.path.join(sparse_reconstruction_dir, 'points.ply')}")
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(sparse_reconstruction_dir, "points.ply"))

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
