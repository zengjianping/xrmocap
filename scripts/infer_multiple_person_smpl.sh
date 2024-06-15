#!/bin/bash

start_frame=0
end_frame=0

image_and_camera_param="xrmocap_data/Shelf_50/image_and_camera_param.txt"
output_dir="output/estimation/Shelf_50/result_is"

image_and_camera_param="xrmocap_data/ezgolf/golf_20240418/image_and_videos/2024-04-17_17-47-34/video_and_camera_param.txt"
output_dir="output/estimation/ezgolf/2024-04-17_17-47-34/result_vs"

#image_and_camera_param="xrmocap_data/panoptic/image_and_videos/dance2a/image_and_camera_param.txt"
#output_dir="output/estimation/panoptic/dance2a/result_is"

python tools/mview_mperson_smpl_estimator.py \
	--estimator_config 'configs/golfpose/mview_mperson_smpl_estimator_s.py' \
	--image_and_camera_param "$image_and_camera_param" \
	--output_dir "$output_dir" \
	--start_frame $start_frame --end_frame $end_frame \
	--enable_log_file
