#!/bin/bash

work_mode=0
width=1124
height=1024

input="xrmocap_data/ezgolf/golf_20240418/camera_parameters/golf_swing"
output="xrmocap_data/ezgolf/golf_20240418/camera_parameters"

#input="xrmocap_data/panoptic/camera_parameters/cameras"
#output="xrmocap_data/panoptic/camera_parameters"

#input="xrmocap_data/panoptic/camera_parameters/calibration.json"
#output="xrmocap_data/panoptic/camera_parameters"

python tools/prepare_camera.py \
	--work_mode $work_mode --width $width --height $height \
	--input $input --output $output

