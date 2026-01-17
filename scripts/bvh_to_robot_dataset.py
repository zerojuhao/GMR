import argparse
import pathlib
import os
import mujoco as mj
import numpy as np
from tqdm import tqdm
import torch
import pickle

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from rich import print


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        default="/home/msi/Desktop/lafan1",
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="/home/msi/Desktop/lafan1_tgt"
    )
    
    parser.add_argument(
        "--robot",
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--target_fps",
        default=50,
        type=int,
    )

    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

   
   
        
    # walk over all files in src_folder
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in tqdm(sorted(filenames), desc="Retargeting files"):
            if not filename.endswith(".bvh"):
                continue
                
            # get the bvh file path
            bvh_file_path = os.path.join(dirpath, filename)
            
            # get the target file path
            tgt_file_path = bvh_file_path.replace(src_folder, tgt_folder).replace(".bvh", ".pkl")

            if os.path.exists(tgt_file_path) and not args.override:
                print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
                continue
            
            # Load LAFAN1 trajectory
            try:
                lafan1_data_frames, actual_human_height = load_bvh_file(bvh_file_path)
                src_fps = 30  # LAFAN1 data is typically 30 FPS
            except Exception as e:
                print(f"Error loading {bvh_file_path}: {e}")
                continue

            
            # Initialize the retargeting system
            retarget = GMR(
                src_human="bvh_lafan1",
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )
            model = mj.MjModel.from_xml_path(retarget.xml_file)
            data = mj.MjData(model)

            

            # retarget to get all qpos
            qpos_list = []
            keypoint_names = None
            keypoints = []
            for curr_frame in range(len(lafan1_data_frames)):
                smplx_data = lafan1_data_frames[curr_frame]
                
                # Retarget till convergence
                qpos = retarget.retarget(smplx_data)
                
                qpos_list.append(qpos.copy())
                
                # 堆叠 scaled_human_data 为 (帧, 关键点, 7) 的 numpy 数组
                # 假设每帧为 dict: {body: (pos, quat)}
                if keypoint_names is None:
                    keypoint_names = list(retarget.scaled_human_data.keys())
                    num_keypoints = len(keypoint_names)
                # 按 key 顺序堆叠
                frame_kps = []
                for name in keypoint_names:
                    pos, quat = retarget.scaled_human_data[name]
                    frame_kps.append(np.concatenate([np.asarray(pos).reshape(3), np.asarray(quat).reshape(4)]))
                keypoints.append(np.stack(frame_kps, axis=0))
                # 在循环外部（如全部帧处理完后）可用：
                # keypoints_arr = np.stack(keypoints, axis=0)  # shape=(帧, 关键点, 7)
            
            qpos_list = np.array(qpos_list)

            # Initialize the forward kinematics
            device = "cuda:0"
            kinematics_model = KinematicsModel(retarget.xml_file, device=device)
            
            root_pos = qpos_list[:, :3]
            root_rot = qpos_list[:, 3:7]
            root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
            dof_pos = qpos_list[:, 7:]
            num_frames = root_pos.shape[0]
            
            # obtain local body pos
            identity_root_pos = torch.zeros((num_frames, 3), device=device)
            identity_root_rot = torch.zeros((num_frames, 4), device=device)
            identity_root_rot[:, -1] = 1.0
            local_body_pos, _ = kinematics_model.forward_kinematics(
                identity_root_pos, 
                identity_root_rot, 
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
            )
            body_names = kinematics_model.body_names
            dof_names = kinematics_model.dof_names
            
            HEIGHT_ADJUST = True
            PERFRAME_ADJUST = True
            if HEIGHT_ADJUST:
                body_pos, _ = kinematics_model.forward_kinematics(
                    torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                    torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                    torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
                )
                ground_offset = 0.00
                if not PERFRAME_ADJUST:
                    lowest_height = torch.min(body_pos[..., 2]).item()
                    root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
                else:
                    for i in range(root_pos.shape[0]):
                        lowest_body_part = torch.min(body_pos[i, :, 2])
                        root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset

            # --- 30Hz -> 50Hz 插值 ---
            target_fps = args.target_fps
            if src_fps != target_fps:
                from scipy.interpolate import interp1d
                from scipy.spatial.transform import Rotation as R, Slerp
                old_num = qpos_list.shape[0]
                new_num = int(np.round(old_num * target_fps / src_fps))
                old_times = np.linspace(0, old_num - 1, old_num)
                new_times = np.linspace(0, old_num - 1, new_num)
                # root_pos插值
                root_pos_interp = interp1d(old_times, root_pos, axis=0, kind='linear')
                root_pos = root_pos_interp(new_times)
                # dof_pos插值
                dof_pos_interp = interp1d(old_times, dof_pos, axis=0, kind='linear')
                dof_pos = dof_pos_interp(new_times)
                # root_rot四元数slerp插值
                r_root = R.from_quat(root_rot)
                slerp_root = Slerp(old_times, r_root)
                root_rot = slerp_root(new_times).as_quat()
                # 关键点插值
                keypoints_arr = np.stack(keypoints, axis=0)  # (old_num, n_kp, 7)
                # xyz线性插值
                xyz_interp = interp1d(old_times, keypoints_arr[..., :3], axis=0, kind='linear')
                xyz_new = xyz_interp(new_times)
                # 四元数slerp插值
                quat_old = keypoints_arr[..., 3:7]  # (old_num, n_kp, 4)
                quat_new = np.zeros((new_num, quat_old.shape[1], 4))
                for j in range(quat_old.shape[1]):
                    r = R.from_quat(quat_old[:, j, :])
                    slerp = Slerp(old_times, r)
                    quat_new[:, j, :] = slerp(new_times).as_quat()
                # 拼接
                keypoints_arr = np.concatenate([xyz_new, quat_new], axis=-1)
                num_frames = new_num
                print(f"Resampled from {old_num} to {new_num} frames for {src_fps}Hz -> {target_fps}Hz.")
                src_fps = target_fps
                
            # motion_data = {
            #     "root_pos": root_pos,
            #     "root_rot": root_rot,
            #     "dof_pos": dof_pos,
            #     "local_body_pos": local_body_pos.detach().cpu().numpy(),
            #     "fps": src_fps,
            #     "link_body_list": body_names,
            # }

            dof_vel = np.vstack([
                    np.zeros((1, dof_pos.shape[1]), dtype=dof_pos.dtype),
                    (dof_pos[1:] - dof_pos[:-1]) * target_fps
                ])
            # 保存关键点数据为 (帧, 关键点, 7) 的 numpy 数组
            keypoint_frames = np.stack(keypoints_arr, axis=0)  # shape=(帧, 关键点, 7)
            key_points_data = {
                # "fps": target_fps,
                # "root_pos": root_pos,
                # "root_rot": root_rot,
                # "dof_pos": dof_pos,
                # "dof_vel": dof_vel,
                # "dof_names": dof_names,
                # "body_names": body_names,
                # "keypoint_names": keypoint_names,
                "keypoint_pos": keypoint_frames[:, :, :3],
                "keypoint_rot": keypoint_frames[:, :, 3:7],
            }

            # 合并保存
            # save_dict = {"key_points_data": key_points_data}

            os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
            with open(tgt_file_path, "wb") as f:
                pickle.dump(key_points_data, f)

            # 打印详细shape信息
            # print("motion_data shapes:")
            # for k, v in motion_data.items():
            #     if hasattr(v, 'shape'):
            #         print(f"  {k}: {v.shape}")
            #     else:
            #         print(f"  {k}: {type(v)}")
            print("key_points_data shapes:")
            for k, v in key_points_data.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
            # print("keypoint_names:", keypoint_names)
    print("Done. saved to ", tgt_folder)
