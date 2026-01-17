import argparse
from email.policy import default
import pathlib
import os
import time

import numpy as np
from sympy import false, true

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel
import torch
import sys
import select
import termios
import tty
import joblib

from rich import print


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        
        # stand 1
        # default="/home/msi/Desktop/ACCAD/Male2General_c3d/A1-_Stand_stageii.npz", # 站立不动
        
        # male2 walk 9
        # default="/home/msi/Desktop/ACCAD/Male2Walking_c3d/B4_-_Stand_to_Walk_backwards_stageii.npz",
        # default="/home/msi/Desktop/ACCAD/Male2Walking_c3d/B9_-__Walk_turn_left_90_stageii.npz", # 250-450
        # default="/home/msi/Desktop/ACCAD/Male2Walking_c3d/B10_-__Walk_turn_left_45_stageii.npz", # 300-500
        # default="/home/msi/Desktop/ACCAD/Male2Walking_c3d/B13_-__Walk_turn_right_90_stageii.npz", # 200-450
        # default="/home/msi/Desktop/ACCAD/Male2Walking_c3d/B14_-__Walk_turn_right_45_t2_stageii.npz", # 300-550
        # default="/home/msi/Desktop/ACCAD/Male2Walking_c3d/B15_-__Walk_turn_around_stageii.npz", # 0-300
        # default="/home/msi/Desktop/ACCAD/Male2Walking_c3d/B22_-__side_step_left_stageii.npz",
        # default="/home/msi/Desktop/ACCAD/Male2Walking_c3d/B23_-__side_step_right_stageii.npz",
        
        # default="/home/msi/Desktop/ACCAD/Male2Walking_c3d/B5_-__Walk_backwards_stageii.npz", # 需要截断开头的数据
        # default="/home/msi/Desktop/ACCAD/Male2Walking_c3d/B11_-__Walk_turn_left_135_stageii.npz", # 需要截断末尾的数据
        


        # male2 run 8
        # default="/home/msi/Desktop/ACCAD/Male2Running_c3d/C1_-_stand_to_run_stageii.npz",
        # default="/home/msi/Desktop/ACCAD/Male2Running_c3d/C3_-_run_stageii.npz",
        # default="/home/msi/Desktop/ACCAD/Male2Running_c3d/C4_-_run_to_walk_a_stageii.npz",
        # default="/home/msi/Desktop/ACCAD/Male2Running_c3d/C4_-_run_to_walk_stageii.npz",
        # default="/home/msi/Desktop/ACCAD/Male2Running_c3d/C5_-_walk_to_run_stageii.npz",
        # default="/home/msi/Desktop/ACCAD/Male2Running_c3d/C12_-_run_turn_left_45_stageii.npz", # 80-230
        # default="/home/msi/Desktop/ACCAD/Male2Running_c3d/C15_-_run_turn_right_45_stageii.npz",
        # default="/home/msi/Desktop/ACCAD/Male2Running_c3d/C17_-_run_change_direction_stageii.npz", # 50-200
        
        # default="/home/msi/Desktop/ACCAD/Male2MartialArtsStances_c3d/D2_-_walk_to_ready_stageii.npz",

        
        # CMU motion
        # default="/home/msi/Desktop/CMU/02/02_02.npz", # walk
        # default="/home/msi/Desktop/CMU/16/16_34.npz", # walk stop

        
        # default="/home/msi/Desktop/CMU/36/36_02_stageii.npz", # 走路 右转慢转身
        # default="/home/msi/Desktop/CMU/36/36_03_stageii.npz", # 走路 右转慢转身
        # default="/home/msi/Desktop/CMU/36/36_09_stageii.npz", # 小步走 跨步 慢转身
        
        # default="/home/msi/Desktop/CMU/127/127_15_stageii.npz", # 跑步 左转 停下
        # default="/home/msi/Desktop/CMU/127/127_16_stageii.npz", # 跑步 右转 停下
        
        # default="/home/msi/Desktop/CMU/127/127_03.npz", # stand to 跑步
        # default="/home/msi/Desktop/CMU/127/127_04.npz", # walk to run
        
        default="/home/msi/Desktop/CMU/127/127_06.npz", # run
        # default="/home/msi/Desktop/CMU/127/127_09.npz", # Run Right
        # default="/home/msi/Desktop/CMU/127/127_11.npz", # Run Left
        # default="/home/msi/Desktop/CMU/127/127_15_stageii.npz", # Run Left
        # default="/home/msi/Desktop/CMU/127/127_16_stageii.npz", # Run Right
        # default="/home/msi/Desktop/CMU/127/127_18.npz", # Run Stop Run
        # default="/home/msi/Desktop/CMU/143/143_02.npz", # Run to Stop
        # default="/home/msi/Desktop/CMU/143/143_03.npz", # Start to Run
        
        
        # default="/home/msi/Desktop/CMU/88/88_05.npz", # 侧翻+空翻
        # default="/home/msi/Desktop/ACCAD/Female1Gestures_c3d/D6-_CartWheel_stageii.npz", 
        
        # default="/home/msi/Desktop/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz",
        # default="/home/msi/Desktop/BMLhandball/S01_ExpertXXXX/Trial_upper_left_033_poses.npz",
        
        
        )
   
    parser.add_argument(
        "--save_slice",
        default=False, # True or False
        help="Whether to save a slice of the robot motion.",
    )

    parser.add_argument(
        "--slice_motion_start_end",
        default=[80, 230],
        help="Whether to save a slice of the robot motion.",
    )

    parser.add_argument(
        "--save_as_pkl",
        default=True, # True or False
        help="whether to save the robot motion as pkl format.",
    )

    parser.add_argument(
        "--save_as_txt",
        default=False, # True or False
        help="whether to save the robot motion as txt format.",
    )
    
    parser.add_argument(
        "--save_as_csv", 
        default=False, # True or False
        help="whether to save the robot motion as csv format.",
    )
    
    parser.add_argument(
        "--save_as_npz",
        default=False, # True or False
        help="whether to save the robot motion as npz format.",
    )

    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openlong", "roboparty_atom01", "roboparty_atom01_long_base_link","roboparty_atom02", "atom01msver"],
        # default="roboparty_atom01",
        # default="roboparty_atom01_long_base_link",
        default="roboparty_atom02",
        # default="unitree_g1",
        # default="atom01msver",
    )
    args_cli = parser.parse_args()
    parser.add_argument(
        "--save_path",
        default=f"{args_cli.robot}_tgt",
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=True,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )



    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    src_fps = smplx_data["mocap_frame_rate"].item()
    tgt_fps = src_fps
    print("原数据集FPS:", src_fps)
    print("目标FPS:", tgt_fps)
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize the retargeting system
    retargeter = GMR(
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)


    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    
    qpos_list = []
            
    # Start the viewer 可导出数据集切片 按空格可暂停播放

    # from 36-02: 150-290右转  650-720左转 900-1030右转
    # from 36-03: 180-350右转  600-750左转 1100-1300右转 1520-1700左转
    # from 36-09: 450-580右转  1050-1200左转 1450-1600左转
    START_FRAME = args.slice_motion_start_end[0] # e.g., 180
    END_FRAME = args.slice_motion_start_end[1]

    # False 则保存全部帧
    SAVE_SLICE = args.save_slice
    if SAVE_SLICE == True:
        if START_FRAME >= len(smplx_data_frames):
            print(f"START_FRAME {START_FRAME} exceeds total frames {len(smplx_data_frames)}. Adjusting to {len(smplx_data_frames)-1}.")
            START_FRAME = len(smplx_data_frames) - 1
        if END_FRAME > len(smplx_data_frames):
            print(f"END_FRAME {END_FRAME} exceeds total frames {len(smplx_data_frames)}. Adjusting to {len(smplx_data_frames)}.")
            END_FRAME = len(smplx_data_frames)
        i = START_FRAME
        n_frames_total = END_FRAME
    else:
        i = 0
        n_frames_total = len(smplx_data_frames)
        
        
# ...existing code...
    # --- 在终端设置 cbreak，按空格切换 paused ---
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    paused = False

    try:
        while True:
            # 非阻塞读取按键
            dr, _, _ = select.select([sys.stdin], [], [], 0.0)
            if dr:
                ch = sys.stdin.read(1)
                if ch == " ":
                    paused = not paused
                    print(f"Paused: {paused}")
                elif ch in ("\x03", "q"):
                    print("Exit requested.")
                    break

            if paused:
                time.sleep(0.05)
                continue

            if args.loop:
                i = (i + 1) % n_frames_total
            else:
                i += 1
                if i >= n_frames_total:
                    break
            print(f"Processing frame {i+1}/{n_frames_total}")
            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

            # Update task targets.
            smplx_data = smplx_data_frames[i]

            # retarget
            qpos = retargeter.retarget(smplx_data)

            # 获取机器人对应连杆名称列表
            robot_frames = retargeter.ik_match_table1.keys()
        
            # qpos[26] = qpos[26] + 0.1
            # qpos[21] = qpos[21] - 0.1
            # qpos[25] = qpos[25] - 0.1
            # qpos[20] = qpos[20] - 0.1

            # visualize
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                # human_motion_data=smplx_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                robot_frames=robot_frames,
                show_robot_body_name=True,
                rate_limit=args.rate_limit,
            )

            if args.save_path is not None:
                qpos_list.append(qpos)
    finally:
        # 恢复终端设置
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
# ...existing code...
    
    
    robot_motion_viewer.close()
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=False,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    
    
    device = "cuda"  # 使用CPU 或 cuda
    kinematics_model = KinematicsModel(retargeter.xml_file, device=device)

    # Ensure qpos_list is a numpy array before slicing (fix TypeError)
    if 'qpos_list' in locals() and isinstance(qpos_list, list):
        qpos_list = np.array(qpos_list)
        
    if SAVE_SLICE:
        print(f"Saved frames from {START_FRAME} to {END_FRAME}, total {len(qpos_list)} frames.")
    
    
    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]] # xyzw to wxyz
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]

    body_names = kinematics_model.body_names
    dof_names = kinematics_model.dof_names

    HEIGHT_ADJUST = True
    if HEIGHT_ADJUST:
        # height adjust to ensure the lowerset part is on the ground
        body_pos, body_rot = kinematics_model.forward_kinematics(torch.from_numpy(root_pos).to(device=device, dtype=torch.float), 
                                                        torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
                                                        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)) # TxNx3
        ground_offset = -0.05
        # ground_offset = torch.min(body_pos[..., 2]).item()
        # root_pos[:, 2] = root_pos[:, 2] - lowerst_height + ground_offset # make sure motion on the ground
        root_pos[:, 2] = root_pos[:, 2] - ground_offset # make sure motion on the ground
        
    ROOT_ORIGIN_OFFSET = True
    if ROOT_ORIGIN_OFFSET:
        # offset using the first frame
        root_pos[:, :2] -= root_pos[0, :2]


    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )


    print("Generating visualization with adjusted motion...")
    i = 0
    n_frames_total = len(qpos_list)

    
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds

    # --- 在终端设置 cbreak，按空格切换 paused ---
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    paused = False
    root_rot[:, [1, 2, 3, 0]] = root_rot[:, [0, 1, 2, 3]] # wxyz to xyzw
    
    try:
        while True:
            # 非阻塞读取按键
            dr, _, _ = select.select([sys.stdin], [], [], 0.0)
            if dr:
                ch = sys.stdin.read(1)
                if ch == " ":
                    paused = not paused
                    print(f"Paused: {paused}")
                elif ch in ("\x03", "q"):
                    print("Exit requested.")
                    break

            if paused:
                time.sleep(0.05)
                continue

            if args.loop:
                i = (i + 1) % (n_frames_total-1)
            else:
                i += 1
                if i >= (n_frames_total-1):
                    break
            print(f"Processing frame {i+1}/{n_frames_total-1}")
            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

            # Update task targets.
            smplx_data = smplx_data_frames[i]

            # visualize
            robot_motion_viewer.step(
                root_pos=root_pos[i],
                root_rot=root_rot[i],
                dof_pos=dof_pos[i],
                human_motion_data=retargeter.scaled_human_data,
                # human_motion_data=smplx_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
            )
            

    finally:
        # 恢复终端设置
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
# ...existing code...

    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]] # xyzw to wxyz

    if args.save_path is not None:
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_names": dof_names,
            "body_names": body_names,
            # "link_body_list": body_names, # # to be modified if needed in AMP(legged_lab)
            "dof_positions": dof_pos,
            "dof_pos": dof_pos,
            "body_positions": body_pos,
            "body_rotations": body_rot,
            "local_body_pos": body_pos,
        }
        print("saving motion data...")
        print("body_names:", body_names)
        print("dof_names:", dof_names)

            
        # helpers for saving in different formats
        def to_numpy_compatible(d):
            """Return a dict with numpy arrays / python scalars suitable for np.savez."""
            out = {}
            for k, v in d.items():
                # torch tensor -> numpy
                if isinstance(v, torch.Tensor):
                    out[k] = v.detach().cpu().numpy()
                # numpy array -> keep
                elif isinstance(v, np.ndarray):
                    out[k] = v
                # list/tuple -> convert to numpy
                elif isinstance(v, (list, tuple)):
                    try:
                        out[k] = np.array(v)
                    except Exception:
                        out[k] = np.array([str(v)])
                # basic types (int/float/str/... including numpy scalar)
                elif isinstance(v, (np.generic,)):
                    try:
                        out[k] = np.array(v).tolist()
                    except Exception:
                        out[k] = v
                else:
                    out[k] = v
            return out

        base_name = os.path.splitext(os.path.basename(args.smplx_file))[0]
        base_no_ext = os.path.join(args.save_path, base_name)
        npz_path = base_no_ext + ".npz"
        pkl_path = base_no_ext + ".pkl"
        txt_path = base_no_ext + ".txt"
        npz_dir = os.path.dirname(npz_path)
        if npz_dir:
            os.makedirs(npz_dir, exist_ok=True)
        # 准备 numpy-compatible dict
        try:
            npz_dict = to_numpy_compatible(motion_data)
        except Exception as e:
            print(f"[ERROR] Converting to numpy-compatible failed for {npz_path}: {e}")
            npz_dict = {}
            
        # 选择保存格式
        PKL = args.save_as_pkl
        NPZ = args.save_as_npz
        TXT = args.save_as_txt
        CSV = args.save_as_csv
    
        if NPZ:
            # 1) 保存 npz
            try:
                np.savez_compressed(npz_path, **npz_dict)
                print(f"Saved to {npz_path}")
            except Exception as e:
                print(f"[ERROR] Saving .npz failed for {npz_path}: {e}")
                
        if PKL:
            # 2) 保存 pkl
            try:
                joblib.dump(npz_dict, pkl_path)
                print(f"Saved to {pkl_path}")
            except Exception as _e:
                print(f"[WARN] joblib dump failed for {pkl_path}: {_e}")

        if TXT:
            # 3) 保存 txt
            try:
                with open(txt_path, "w") as f:
                    for k, v in npz_dict.items():
                        f.write(f"{k}: {v}\n")
                print(f"Saved to {txt_path}")
            except Exception as e:
                print(f"[ERROR] Saving .txt failed for {txt_path}: {e}")

        if CSV:
            # 4) 保存 csv
            try:
                def export_to_csv(root_pos, root_rot, dof_pos, filename):
                    num_frames = root_pos.shape[0]
                    with open(filename, 'w') as f:
                        for i in range(num_frames):
                            row = [f"{root_pos[i, j]:.6f}" for j in range(3)]
                            row += [f"{root_rot[i, j]:.6f}" for j in range(4)]
                            row += [f"{dof_pos[i, j]:.6f}" for j in range(dof_pos.shape[1])]
                            f.write(','.join(row) + '\n')
                csv_path = base_no_ext + ".csv"
                export_to_csv(root_pos, root_rot, dof_pos, csv_path) # 8 14
                print(f"Saved to {csv_path}")
                
            except Exception as e:
                print(f"[ERROR] Saving .csv failed for {csv_path}: {e}")

    print("data shape:", {k: np.shape(v) for k, v in npz_dict.items()})
    robot_motion_viewer.close()
