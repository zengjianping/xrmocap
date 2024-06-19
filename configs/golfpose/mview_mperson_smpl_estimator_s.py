type = 'MultiViewMultiPersonSMPLEstimator'
bbox_thr = 0.4
work_dir = './temp'
verbose = True
logger = None
pred_kps3d_convention = 'golfpose'
optimize_kps3d = True
output_smpl = False
multi_person = False

if False:
    bbox_detector = dict(
        type='MMdetDetector', batch_size=10,
        mmdet_kwargs=dict(device='cuda',
            checkpoint='weight/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
            config='configs/modules/human_perception/mmdet_faster_rcnn_r50_fpn_coco.py')
    )
    kps2d_estimator = dict(
        type='MMposeTopDownEstimator', bbox_thr=bbox_thr,
        mmpose_kwargs=dict(device='cuda',
            checkpoint='weight/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth',
            config='configs/modules/human_perception/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py')
    )
elif True:
    bbox_detector = dict(
        type='MMdetDetector', batch_size=10, bbox_thr=bbox_thr,
        mmdet_kwargs=dict(device='cuda', palette='random',
            checkpoint='datas/models/golfdet/v1/epoch_70.pth',
            config='datas/models/golfdet/v1/rtmdet_tiny_8xb32-300e_golfpose.py')
    )
    kps2d_estimator = dict(
        type='MMposeTopDownEstimator', bbox_thr=bbox_thr,
        mmpose_kwargs=dict(device='cuda',
            checkpoint='datas/models/golfpose/v2/best_AUC_epoch_100.pth',
            config='datas/models/golfpose/v2/rtmpose-m_golfpose-256x192.py')
    )
elif True:
    bbox_detector = dict(
        type='MMdetDetector', batch_size=10,
        mmdet_kwargs=dict(device='cuda',
            checkpoint='weight/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
            config='configs/modules/human_perception/rtmdet_m_640-8xb32_coco-person.py')
    )
    kps2d_estimator = dict(
        type='MMposeTopDownEstimator', bbox_thr=bbox_thr,
        mmpose_kwargs=dict(device='cuda',
            checkpoint='weight/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
            config='configs/modules/human_perception/rtmpose-m_8xb256-420e_body8-256x192.py')
    )
else:
    bbox_detector = dict(
        type='MMdetTrtDetector', device='cuda', batch_size=10, 
        deploy_cfg='configs/modules/human_perception/deploy/detection_tensorrt_dynamic-320x320-1344x1344.py',
        model_cfg='configs/modules/human_perception/mmdet_faster_rcnn_r50_fpn_coco.py',
        backend_files=['weight/mmdet_faster_rcnn/end2end.engine']
    )
    kps2d_estimator = dict(
        type='MMposeTrtTopDownEstimator', device='cuda', bbox_thr=0.95,
        deploy_cfg='configs/modules/human_perception/deploy/pose-detection_tensorrt_dynamic-384x288.py',
        model_cfg='configs/modules/human_perception/mmpose_hrnet_w48_coco_wholebody_384x288_dark_plus.py',
        backend_files=['weight/mmpose_hrnet/end2end.engine']
    )

associator = dict(
    type='MvposeAssociator',
    triangulator=dict(
        type='AniposelibTriangulator',
        camera_parameters=[],
        logger=logger,
    ),
    affinity_estimator=dict(type='AppearanceAffinityEstimator', init_cfg=None),
    #point_selector=dict(type='HybridKps2dSelector', convention=pred_kps3d_convention,
    #    triangulator=dict(type='AniposelibTriangulator', camera_parameters=[], logger=logger),
    #    ignore_kps_name=['left_eye', 'right_eye', 'left_ear', 'right_ear'], verbose=False),
    #point_selector=dict(type='ManualThresholdSelector', threshold=0.0, verbose=verbose),
    point_selector = dict(type='CameraErrorSelector', target_camera_number=2, verbose=verbose,
        triangulator=dict(type='AniposelibTriangulator', camera_parameters=[])),
    #point_selector = dict(type='ReprojectionErrorPointSelectorEx', target_camera_number=3,
    #    triangulator=dict(type='AniposelibTriangulator', camera_parameters=[]), verbose=verbose),
    multi_way_matching=dict(
        type='MultiWayMatching',
        use_dual_stochastic_SVT=True,
        lambda_SVT=50,
        alpha_SVT=0.0001,
        n_cam_min=2,
    ),
    kalman_tracking=dict(type='KalmanTracking', n_cam_min=3, logger=logger),
    identity_tracking=dict(
        type='KeypointsDistanceTracking',
        tracking_distance=100,
        tracking_kps3d_convention=pred_kps3d_convention,
        tracking_kps3d_name=[
            'left_shoulder', 'right_shoulder', 'left_hip_extra',
            'right_hip_extra'
        ]),
    checkpoint_path='./weight/mvpose/resnet50_reid_camstyle-98d61e41_20220921.pth',
    best_distance=800,
    interval=5,
    bbox_thr=bbox_thr,
    device='cuda',
    logger=logger,
)

triangulator = dict(
    type='AniposelibTriangulator',
    camera_parameters=[],
    logger=logger,
)
#cam_pre_selector = dict(
#    type='ManualThresholdSelector', threshold=0.4, verbose=verbose)
#cam_selector = dict(
#    type='CameraErrorSelector',
#    target_camera_number=3,
#    triangulator=dict(type='AniposelibTriangulator', camera_parameters=[]),
#    verbose=verbose)
point_selectors = [
    dict(type='ManualThresholdSelector', threshold=0.0, verbose=verbose),
    #dict(type='AutoThresholdSelector', start=0.95, stride=-0.025, verbose=verbose)
    dict(type='ReprojectionErrorPointSelectorEx', target_camera_number=3,
         triangulator=dict(type='AniposelibTriangulator', camera_parameters=[]),
         tolerance_error=100, verbose=verbose)
]

kps3d_optimizers = [
    dict(type='NanInterpolation', verbose=verbose),
    dict(type='AniposelibOptimizerEx', smooth_weight=2.0, verbose=verbose,
         triangulator=dict(type='AniposelibTriangulator', camera_parameters=[]))
]

smplify = dict(
    type='SMPLify',
    verbose=verbose,
    info_level='stage',
    n_epochs=1,
    use_one_betas_per_video=True,
    hooks=[
        dict(type='SMPLifyVerboseHook'),
    ],
    logger=logger,
    body_model=dict(
        type='SMPL',
        gender='neutral',
        num_betas=10,
        keypoint_convention='smpl_45',
        model_path='xrmocap_data/body_models/smpl',
        batch_size=1,
        logger=logger),
    optimizer = dict(
        type='OptimWrapper', # Specify the type of OptimWrapper
        optimizer=dict(type='LBFGS', max_iter=20, lr=1.0, line_search_fn='strong_wolfe')),
    ignore_keypoints=[
        'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
        'right_hip_extra', 'left_hip_extra'
    ],
    handlers=[
        dict(
            handler_key='keypoints3d_mse',
            type='Keypoint3dMSEHandler',
            mse_loss=dict(
                type='KeypointMSELoss',
                loss_weight=10.0,
                reduction='sum',
                sigma=100),
            logger=logger),
        dict(
            handler_key='keypoints2d_mse',
            type='MultiviewKeypoint2dMSEHandler',
            mse_loss=dict(
                type='KeypointMSELoss',
                loss_weight=1.0,
                reduction='sum',
                sigma=100),
            logger=logger),
        dict(
            handler_key='shape_prior',
            type='BetasPriorHandler',
            prior_loss=dict(
                type='ShapePriorLoss', loss_weight=5e-3, reduction='mean'),
            logger=logger),
        dict(
            handler_key='joint_prior',
            type='BodyPosePriorHandler',
            prior_loss=dict(
                type='JointPriorLoss',
                loss_weight=20.0,
                reduction='sum',
                smooth_spine=True,
                smooth_spine_loss_weight=20,
                use_full_body=True),
            logger=logger),
        dict(
            handler_key='smooth_joint',
            type='BodyPosePriorHandler',
            prior_loss=dict(
                type='SmoothJointLoss',
                loss_weight=1.0,
                reduction='mean',
                loss_func='L2'),
            logger=logger),
        dict(
            handler_key='pose_prior',
            type='BodyPosePriorHandler',
            prior_loss=dict(
                type='MaxMixturePriorLoss',
                prior_folder='xrmocap_data/body_models',
                num_gaussians=8,
                loss_weight=4.78**2,
                reduction='sum'),
            logger=logger),
        dict(
            handler_key='pose_reg',
            type='BodyPosePriorHandler',
            prior_loss=dict(
                type='PoseRegLoss', loss_weight=0.001, reduction='mean'),
            logger=logger),
        dict(
            handler_key='keypoints3d_limb_len',
            type='Keypoint3dLimbLenHandler',
            loss=dict(
                type='LimbLengthLoss',
                convention='smpl',
                loss_weight=1.0,
                reduction='mean'),
            logger=logger),
    ],
    stages=[
        # stage 0
        dict(
            n_iter=10,
            ftol=1e-4,
            fit_global_orient=False,
            fit_transl=False,
            fit_body_pose=False,
            fit_betas=True,
            keypoints3d_mse_weight=0.0,
            keypoints2d_mse_weight=0.0,
            keypoints3d_limb_len_weight=1.0,
            shape_prior_weight=5e-3,
            joint_prior_weight=0.0,
            smooth_joint_weight=0.0,
            pose_reg_weight=0.0,
            pose_prior_weight=0.0),
        # stage 1
        dict(
            n_iter=50,
            ftol=1e-4,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=False,
            fit_betas=False,
            keypoints3d_mse_weight=1.0,
            keypoints2d_mse_weight=1.0,
            keypoints3d_limb_len_weight=0.0,
            shape_prior_weight=0.0,
            joint_prior_weight=0.0,
            smooth_joint_weight=0.0,
            pose_reg_weight=0.0,
            pose_prior_weight=0.0,
            body_weight=0.0,
            face_weight=0.0,
            hand_weight=0.0,
            shoulder_weight=1.0,
            hip_weight=1.0,
            use_shoulder_hip_only=True),
        # stage 2
        dict(
            n_iter=120,
            ftol=1e-4,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=True,
            fit_betas=False,
            keypoints3d_mse_weight=10.0,
            keypoints2d_mse_weight=0.0,
            keypoints3d_limb_len_weight=0.0,
            shape_prior_weight=0.0,
            joint_prior_weight=0.0,
            smooth_joint_weight=1.0,
            pose_reg_weight=0.001,
            pose_prior_weight=0.0,
            body_weight=1.0,
            use_shoulder_hip_only=False),
    ],
)
