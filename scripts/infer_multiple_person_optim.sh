#!/bin/bash

python tools/mview_mperson_topdown_estimator.py \
	--estimator_config 'configs/mvpose_tracking/mview_mperson_topdown_estimator.py' \
	--image_and_camera_param 'xrmocap_data/Shelf_50/image_and_camera_param.txt' \
	--start_frame 300 \
	--end_frame 350 \
	--output_dir 'output/estimation/Shelf_50' \
	--enable_log_file
