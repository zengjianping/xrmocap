import os, sys, cv2
import argparse, logging
import shutil, glob, json
import numpy as np

from mmhuman3d.utils.demo_utils import get_different_colors
from xrprimer.data_structure.camera import FisheyeCameraParameter


def convert_camera_param(input_file, output_file, camera_idx, width, height):
    fs = cv2.FileStorage(input_file, cv2.FILE_STORAGE_READ)
    output_data = dict()
    output_data['class_name'] = "FisheyeCameraParameter"
    output_data['convention'] = "opencv"
    output_data['name'] = f'camera_param_{camera_idx:02d}'
    output_data['width'] = width
    output_data['height'] = height
    output_data['world2cam'] = True
    output_data['extrinsic_r'] = fs.getNode("CameraMatrix").mat()[0:3, 0:3].tolist()
    output_data['extrinsic_t'] = fs.getNode("CameraMatrix").mat()[0:3, 3].T.tolist()
    intrisic_mat = np.zeros((4,4), dtype=np.float32)
    intrisic_mat[0:2, 0:3] = fs.getNode("Intrinsics").mat()[0:2, 0:3]
    intrisic_mat[2, 3] = 1.0
    intrisic_mat[3, 2] = 1.0
    output_data['intrinsic'] = intrisic_mat.tolist()
    distort_mat = fs.getNode("Distortion").mat()
    #distort_mat = np.zeros_like(distort_mat)
    output_data['k1'] = distort_mat[0, 0]
    output_data['k2'] = distort_mat[1, 0]
    output_data['p1'] = distort_mat[2, 0]
    output_data['p2'] = distort_mat[3, 0]
    output_data['k3'] = distort_mat[4, 0]
    if len(distort_mat[:,0]) > 5:
        output_data['k4'] = distort_mat[5, 0]
        output_data['k5'] = distort_mat[6, 0]
        output_data['k6'] = distort_mat[7, 0]
    else:
        output_data['k4'] = 0.0
        output_data['k5'] = 0.0
        output_data['k6'] = 0.0

    json_data = json.dumps(output_data, indent=4, ensure_ascii=False)
    fp = open(output_file, 'w', encoding='utf-8')
    fp.write(json_data)
    fp.close()

def convert_camera_params(input_dir, output_dir, width, height):
    input_files = glob.glob(os.path.join(input_dir, "*.xml"))
    input_files = sorted(input_files, reverse=False)

    for camera_idx, input_file in enumerate(input_files):
        print(f'Processing {camera_idx}/{len(input_files)}...')
        output_file = os.path.join(output_dir, f'camera_param_{camera_idx:02d}.json')
        convert_camera_param(input_file, output_file, camera_idx, width, height)
    
    return True

def convert_panoptic_cameras(input_file, output_dir):
    """Convert source data to XRPrimer camera parameters.

    Args:
        scene_idx (int):
            Index of this scene.
    """

    with open(input_file, 'r') as f_read:
        panoptic_calib_dict = json.load(f_read)
    os.makedirs(output_dir, exist_ok=True)

    for view_idx in range(30):
        fisheye_param = FisheyeCameraParameter(name=f'camera_param_{view_idx:02d}')
        cam_key = f'00_{view_idx:02d}'
        panoptic_cam_dict = None
        for _, dict_value in enumerate(panoptic_calib_dict['cameras']):
            if dict_value['name'] == cam_key:
                panoptic_cam_dict = dict_value
        if panoptic_cam_dict is None:
            continue

        fisheye_param.set_resolution(
            width=panoptic_cam_dict['resolution'][0],
            height=panoptic_cam_dict['resolution'][1])
        translation = np.array(panoptic_cam_dict['t']) / 100.0
        fisheye_param.set_KRT(
            K=panoptic_cam_dict['K'],
            R=panoptic_cam_dict['R'],
            T=translation,
            world2cam=True)
        dist_list = panoptic_cam_dict['distCoef']
        fisheye_param.set_dist_coeff(
            dist_coeff_k=[dist_list[0], dist_list[1], dist_list[4]],
            dist_coeff_p=[dist_list[2], dist_list[3]])

        # dump the distorted camera
        fisheye_param.dump(
            os.path.join(output_dir, f'{fisheye_param.name}.json'))

def main(args):
    if args.work_mode == 0:
        convert_camera_params(args.input, args.output, args.width, args.height)
    elif args.work_mode == 1:
        convert_panoptic_cameras(args.input, args.output)
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare camera data.')
    parser.add_argument('--work_mode', dest='work_mode', type=int, default=0)
    parser.add_argument('--input', dest='input', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--width', dest='width', type=int, default=0)
    parser.add_argument('--height', dest='height', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

