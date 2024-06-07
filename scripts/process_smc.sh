#!/bin/bash

python tools/process_smc.py \
	--estimator_config configs/humman_mocap/mview_sperson_smpl_estimator.py \
	--smc_path xrmocap_data/humman/p000455_a000986.smc \
	--output_dir xrmocap_data/humman/p000455_a000986_output \
	--visualize --frame_file="none"

