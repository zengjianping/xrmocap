# yapf: disable
import argparse
import cv2
import datetime
import glob
import mmcv
import numpy as np
import os
import shutil
from mmhuman3d.core.visualization.visualize_smpl import (
    visualize_smpl_calibration,
)
from mmhuman3d.utils.demo_utils import get_different_colors
from typing import List
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.utils.log_utils import setup_logger

from xrmocap.core.estimation.builder import build_estimator
from xrmocap.visualization.visualize_keypoints3d import visualize_keypoints2d
from xrmocap.visualization.visualize_keypoints3d import visualize_keypoints3d_projected
from xrprimer.utils.ffmpeg_utils import VideoInfoReader, video_to_array
from xrprimer.visualization.keypoints.visualize_keypoints3d import (
    visualize_keypoints3d,
)

# yapf: enable


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.enable_log_file:
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_path = os.path.join(args.output_dir, f'{time_str}.txt')
        logger = setup_logger(logger_name=__name__, logger_path=log_path)
    else:
        logger = None

    # build estimator
    estimator_config = dict(mmcv.Config.fromfile(args.estimator_config))
    estimator_config['logger'] = logger
    smpl_estimator = build_estimator(estimator_config)
    # load camera parameter and images
    image_dirs = []
    video_files = []
    fisheye_param_paths = []
    with open(args.image_and_camera_param, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if i % 2 == 0:
                if line[-4:] == '.mp4':
                    video_files.append(line)
                else:
                    image_dirs.append(line)
            else:
                fisheye_param_paths.append(line)
    fisheye_params = load_camera_parameters(fisheye_param_paths)

    start_frame = args.start_frame
    end_frame = args.end_frame
    mview_img_list = None
    mview_video_list = None

    if len(video_files) > 0:
        mview_video_list = []
        for idx in range(len(fisheye_params)):
            mview_video_list.append(video_files[idx])
    else:
        mview_img_list = []
        for idx in range(len(fisheye_params)):
            sview_img_list = sorted(glob.glob(os.path.join(image_dirs[idx], '*.jpg')))
            if len(sview_img_list) == 0:
                sview_img_list = sorted(glob.glob(os.path.join(image_dirs[idx], '*.png')))
            if end_frame <= 0:
                end_frame = len(sview_img_list)
            sview_img_list = sview_img_list[start_frame:end_frame]
            mview_img_list.append(sview_img_list)
    pred_keypoints2d_list, pred_keypoints3d, smpl_data_list = smpl_estimator.run(
        cam_param=fisheye_params, img_paths=mview_img_list, video_paths=mview_video_list)
    npz_path = os.path.join(args.output_dir, 'pred_keypoints3d.npz')
    pred_keypoints3d.dump(npz_path)

    if smpl_data_list is not None:
        for i, smpl_data in enumerate(smpl_data_list):
            smpl_data.dump(os.path.join(args.output_dir, f'smpl_{i}.npz'))

    # Visualization
    if not args.disable_visualization:
        n_frame = pred_keypoints3d['keypoints'].shape[0]

        if smpl_data_list is not None:
            n_person = len(smpl_data_list)
            colors = get_different_colors(n_person)
            tmp = colors[:, 0].copy()
            colors[:, 0] = colors[:, 2]
            colors[:, 2] = tmp
            full_pose_list = []
            transl_list = []
            betas_list = []
            for smpl_data in smpl_data_list:
                full_pose_list.append(smpl_data['fullpose'][:, np.newaxis])
                transl_list.append(smpl_data['transl'][:, np.newaxis])
                betas_list.append(smpl_data['betas'][:, np.newaxis])
            fullpose = np.concatenate(full_pose_list, axis=1)
            transl = np.concatenate(transl_list, axis=1)
            betas = np.concatenate(betas_list, axis=1)

        body_model_cfg = dict(
            type='SMPL',
            gender='neutral',
            num_betas=10,
            keypoint_src='smpl_45',
            keypoint_dst='smpl',
            model_path='xrmocap_data/body_models',
            batch_size=1)

        # visualize keypoints in a 3D scene
        if True:
            logger.info('Visualizing keypoints3d.')
            visualize_keypoints3d(
                keypoints=pred_keypoints3d,
                output_path=os.path.join(args.output_dir, 'keypoints3d_pred.mp4'),
                overwrite=True,
                plot_axis=True,
                plot_points=True,
                plot_lines=True,
                disable_tqdm=True,
                logger=logger)

        # prepare camera
        for idx, fisheye_param in enumerate(fisheye_params):
            k_np = np.array(fisheye_param.get_intrinsic(3))
            r_np = np.array(fisheye_param.get_extrinsic_r())
            t_np = np.array(fisheye_param.get_extrinsic_t())
            cam_name = fisheye_param.name
            view_name = cam_name.replace('fisheye_param_', '')

            if len(video_files) > 0:
                image_array = video_to_array(video_files[idx],
                    start=0, end=None, logger=logger)
            else:
                image_list = []
                for frame_path in mview_img_list[idx]:
                    image_np = cv2.imread(frame_path)
                    image_list.append(image_np)
                image_array = np.array(image_list)

            kps2d_output_dir = os.path.join(args.output_dir, 'kps2d')
            os.makedirs(kps2d_output_dir, exist_ok=True)
            kps3d_output_dir = os.path.join(args.output_dir, 'kps3d')
            os.makedirs(kps3d_output_dir, exist_ok=True)
            smpl_output_dir = os.path.join(args.output_dir, 'smpl')
            os.makedirs(smpl_output_dir, exist_ok=True)

            visualize_keypoints2d(
                keypoints=pred_keypoints2d_list[idx],
                output_path=os.path.join(kps2d_output_dir, f'{view_name}_kps2d.mp4'),
                background_arr=image_array.copy(),
                overwrite=True)
            
            visualize_keypoints3d_projected(
                keypoints=pred_keypoints3d,
                camera=fisheye_param,
                output_path=os.path.join(kps3d_output_dir, f'{view_name}_kps3d_projected.mp4'),
                background_arr=image_array.copy(),
                overwrite=True)
            
            if smpl_data_list is not None:
                visualize_smpl_calibration(
                    poses=fullpose.reshape(n_frame, n_person, -1),
                    betas=betas,
                    transl=transl,
                    palette=colors,
                    output_path=os.path.join(smpl_output_dir, f'{view_name}_smpl.mp4'),
                    body_model_config=body_model_cfg,
                    K=k_np,
                    R=r_np,
                    T=t_np,
                    image_array=image_array,
                    resolution=(image_array.shape[1], image_array.shape[2]),
                    overwrite=True)


def load_camera_parameters(fisheye_param_paths: List[str]):
    """Load multi-scene fisheye parameters."""
    mview_list = []
    for path in fisheye_param_paths:
        fisheye_param = FisheyeCameraParameter.fromfile(path)
        if fisheye_param.world2cam:
            fisheye_param.inverse_extrinsic()
        mview_list.append(fisheye_param)

    return mview_list


def setup_parser():
    parser = argparse.ArgumentParser(
        description='MultiViewMultiPersonTopDownEstimator')
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving all possible output files.',
        default='./output/estimation')
    parser.add_argument(
        '--estimator_config',
        help='Config file for MultiViewMultiPersonTopDownEstimator.',
        type=str,
        default='configs/modules/core/estimation/'
        'mview_mperson_topdown_estimator.py')
    parser.add_argument(
        '--image_and_camera_param',
        help='A text file contains the image path and the corresponding'
        'camera parameters',
        default='./xrmocap_data/Shelf/image_and_camera_param.txt')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=0)
    parser.add_argument(
        '--enable_log_file',
        action='store_true',
        help='If checked, log will be written as file.',
        default=False)
    parser.add_argument(
        '--disable_visualization',
        action='store_true',
        help='If checked, visualize result.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
