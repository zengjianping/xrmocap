# yapf: disable
import logging
import cv2
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple, Union, overload
from xrprimer.data_structure import Keypoints
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoint_num,
)

from xrmocap.data_structure.body_model import SMPLData, SMPLXData
from xrmocap.model.registrant.builder import SMPLify, build_registrant
from xrmocap.model.registrant.handler.builder import build_handler
from xrmocap.transform.keypoints3d.optim.builder import (
    BaseOptimizer, build_keypoints3d_optimizer,
)
from .base_estimator import BaseEstimator
from xrmocap.human_perception.builder import (
    MMdetDetector, MMposeTopDownEstimator, build_detector,
)
from xrmocap.io.image import (
    get_n_frame_from_mview_src, load_clip_from_mview_src,
)
from xrmocap.model.registrant.builder import SMPLify
from xrmocap.ops.top_down_association.builder import (
    MvposeAssociator, build_top_down_associator,
)
from xrmocap.ops.triangulation.builder import (
    BaseTriangulator, build_triangulator,
)
from xrmocap.ops.triangulation.point_selection.builder import (
    BaseSelector, CameraErrorSelector, build_point_selector,
)
from xrmocap.transform.keypoints3d.optim.builder import BaseOptimizer
from xrmocap.visualization.visualize_keypoints2d import visualize_keypoints2d

# yapf: enable


class MultiViewMultiPersonSMPLEstimator(BaseEstimator):
    """Api for estimating keypoints3d and smpl in a multi-view multi-person
    scene, using optimization-based top-down method."""

    def __init__(self,
                 bbox_thr: float,
                 work_dir: str,
                 bbox_detector: Union[dict, MMdetDetector],
                 kps2d_estimator: Union[dict, MMposeTopDownEstimator],
                 associator: Union[dict, MvposeAssociator],
                 smplify: Union[dict, SMPLify],
                 triangulator: Union[dict, BaseTriangulator],
                 point_selectors: List[Union[dict, BaseSelector, None]] = None,
                 cam_pre_selector: Union[dict, BaseSelector, None] = None,
                 cam_selector: Union[dict, CameraErrorSelector, None] = None,
                 kps3d_optimizers: Union[List[Union[BaseOptimizer, dict]], None] = None,
                 pred_kps3d_convention: str = 'coco',
                 load_batch_size: int = 10,
                 optimize_kps3d: bool = True,
                 output_smpl: bool = True,
                 multi_person: bool = True,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialization of the class.

        Args:
            bbox_thr (float):
                The threshold of the bbox2d.
            work_dir (str):
                Path to the folder for running the api.
                No file in work_dir will be modified or added by
                MultiViewMultiPersonTopDownEstimator.
            bbox_detector (Union[dict, MMdetDetector]):
                A human bbox_detector or its config.
            kps2d_estimator (Union[dict, MMposeTopDownEstimator]):
                A top-down kps2d estimator or its config.
            associator (Union[dict, MvposeAssociator]):
                A MvposeAssociator instance or its config.
            smplify (Union[dict, SMPLify]):
                A SMPLify instance or its config.
            triangulator (Union[dict, BaseTriangulator]):
                A triangulator or its config.
            cam_pre_selector (Union[dict, BaseSelector, None], optional):
                A selector before selecting cameras. If it's given,
                points for camera selection will be filtered.
                Defaults to None.
            cam_selector (Union[dict, CameraErrorSelector, None], optional):
                A camera selector or its config. If it's given, cameras
                will be selected before triangulation.
                Defaults to None.
            point_selectors (List[Union[dict, BaseSelector, None]], optional):
                A point selector or its config. If it's given, points
                will be selected before triangulation.
                Defaults to None.
            kps3d_optimizers (Union[
                    List[Union[BaseOptimizer, dict]], None], optional):
                A list of keypoints3d optimizers or their configs. If given,
                keypoints3d will be optimized by the cascaded final optimizers.
                Defaults to None.
            pred_kps3d_convention (str, optional): Defaults to 'coco'.
            load_batch_size (int, optional):
                How many frames are loaded at the same time. Defaults to 10.
            verbose (bool, optional):
                Whether to print(logger.info) information during estimating.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.bbox_thr = bbox_thr
        self.load_batch_size = load_batch_size
        self.pred_kps3d_convention = pred_kps3d_convention
        self.optimize_kps3d = optimize_kps3d
        self.output_smpl = output_smpl
        self.multi_person = multi_person

        super().__init__(work_dir, verbose, logger)

        if isinstance(smplify, dict):
            smplify['logger'] = logger
            if smplify['type'].lower() == 'smplify':
                self.smpl_data_type = 'smpl'
            elif smplify['type'].lower() == 'smplifyx':
                self.smpl_data_type = 'smplx'
            else:
                self.logger.error('SMPL data type not found.')
                raise TypeError
            self.algo_smplify = build_registrant(smplify)
        else:
            self.algo_smplify = smplify

        if kps3d_optimizers is None:
            self.algo_kps3d_optimizers = None
        else:
            self.algo_kps3d_optimizers = []
            for kps3d_optim in kps3d_optimizers:
                if isinstance(kps3d_optim, dict):
                    kps3d_optim['logger'] = logger
                    kps3d_optim = build_keypoints3d_optimizer(kps3d_optim)
                self.algo_kps3d_optimizers.append(kps3d_optim)

        if isinstance(bbox_detector, dict):
            bbox_detector['logger'] = logger
            self.algo_bbox_detector = build_detector(bbox_detector)
        else:
            self.algo_bbox_detector = bbox_detector

        if isinstance(kps2d_estimator, dict):
            kps2d_estimator['logger'] = logger
            self.algo_kps2d_estimator = build_detector(kps2d_estimator)
        else:
            self.algo_kps2d_estimator = kps2d_estimator

        if isinstance(associator, dict):
            associator['logger'] = logger
            self.algo_associator = build_top_down_associator(associator)
        else:
            self.algo_associator = associator

        if isinstance(triangulator, dict):
            triangulator['logger'] = logger
            self.algo_triangulator = build_triangulator(triangulator)
        else:
            self.algo_triangulator = triangulator

        if isinstance(cam_pre_selector, dict):
            cam_pre_selector['logger'] = logger
            self.algo_cam_pre_selector = build_point_selector(cam_pre_selector)
        else:
            self.algo_cam_pre_selector = cam_pre_selector

        if isinstance(cam_selector, dict):
            cam_selector['logger'] = logger
            cam_selector['triangulator']['camera_parameters'] = \
                self.algo_triangulator.camera_parameters
            self.algo_cam_selector = build_point_selector(cam_selector)
        else:
            self.algo_cam_selector = cam_selector

        if point_selectors is None:
            self.algo_point_selectors = None
        else:
            self.algo_point_selectors = []
            for selector in point_selectors:
                if isinstance(selector, dict):
                    selector['logger'] = logger
                    selector = build_point_selector(selector)
                self.algo_point_selectors.append(selector)

    @overload
    def run(
        self, img_arr: np.ndarray, cam_param: List[FisheyeCameraParameter]
    ) -> Tuple[List[Keypoints], Keypoints, SMPLData]:
        return self.run(cam_param, img_arr=img_arr)

    @overload
    def run(
        self, img_paths: List[List[str]], cam_param: List[FisheyeCameraParameter]
    ) -> Tuple[List[Keypoints], Keypoints, SMPLData]:
        return self.run(cam_param, img_paths=img_paths)

    @overload
    def run(
        self, video_paths: List[str], cam_param: List[FisheyeCameraParameter]
    ) -> Tuple[List[Keypoints], Keypoints, SMPLData]:
        return self.run(cam_param, video_paths=video_paths)

    def run(
        self,
        cam_param: List[FisheyeCameraParameter],
        img_arr: Union[None, np.ndarray] = None,
        img_paths: Union[None, List[List[str]]] = None,
        video_paths: Union[None, List[str]] = None,
    ) -> Tuple[Keypoints, List[SMPLData]]:
        """Run mutli-view multi-person topdown estimator once. run() needs one
        images input among [img_arr, img_paths, video_paths].

        Args:
            cam_param (List[FisheyeCameraParameter]):
                A list of FisheyeCameraParameter instances.
            img_arr (Union[None, np.ndarray], optional):
                A multi-view image array, in shape
                [n_view, n_frame, h, w, c]. Defaults to None.
            img_paths (Union[None, List[List[str]]], optional):
                A nested list of image paths, in shape
                [n_view, n_frame]. Defaults to None.
            video_paths (Union[None, List[str]], optional):
                A list of video paths, each is a view.
                Defaults to None.

        Returns:
            Tuple[Keypoints, List[SMPLData]]:
                A keypoints3d, a list of SMPLData.
        """

        self.logger.info('Processing video human pose...')

        input_list = [img_arr, img_paths, video_paths]
        input_count = 0
        for input_instance in input_list:
            if input_instance is not None:
                input_count += 1
        if input_count > 1:
            self.logger.error('Redundant input!\n' +
                              'Please offer only one among' +
                              ' img_arr, img_paths and video_paths.')
            raise ValueError
        elif input_count < 1:
            self.logger.error('No image input has been found!\n' +
                              'img_arr, img_paths and video_paths are None.')
            raise ValueError

        self.algo_associator.set_cameras(cam_param)
        self.algo_triangulator.set_cameras(cam_param)
        if self.algo_point_selectors is not None:
            for selector in self.algo_point_selectors:
                if hasattr(selector, 'triangulator'):
                    selector.triangulator.set_cameras(cam_param)

        n_frame = get_n_frame_from_mview_src(img_arr, img_paths, video_paths, self.logger)
        n_view = len(cam_param)
        n_kps = get_keypoint_num(convention=self.pred_kps3d_convention)
        pred_kps3d = np.zeros((n_frame, 1, n_kps, 4))
        association_results = [[] for _ in range(n_frame)]
        selected_keypoints2d = []
        max_identity = 0

        self.logger.info('Estimating human 2d&3d keypoints...')
        for frame_idx in range(0, n_frame):
            self.logger.info(f'Processing video frame {frame_idx+1}/{n_frame}...')

            self.logger.info('Loading frame images...')
            mview_batch_arr = load_clip_from_mview_src(
                start_idx=frame_idx,
                end_idx=frame_idx+1,
                img_arr=img_arr,
                img_paths=img_paths,
                video_paths=video_paths,
                logger=self.logger)
            
            self.logger.info('Finished loading frame images.')
            self.logger.info('Estimating 2d keypoints...')

            # Estimate bbox2d and keypoints2d
            bbox2d_list, keypoints2d_list = self.estimate_keypoints2d(mview_batch_arr)

            self.logger.info('Finished estimating 2d keypoints.')
            self.logger.info('Estimating 3d keypoints...')

            max_identity, pred_kps3d = self.estimate_keypoints3d(bbox2d_list,
                keypoints2d_list, mview_batch_arr, max_identity, pred_kps3d,
                selected_keypoints2d, association_results)

            self.logger.info('Finished estimating 3d keypoints.')
            self.logger.info('Finished processing video frame.')

        # Convert array to keypoints instance
        pred_keypoints3d = Keypoints(
            dtype='numpy',
            kps=pred_kps3d,
            mask=pred_kps3d[..., -1] > 0,
            convention=self.pred_kps3d_convention,
            logger=self.logger)

        # Save keypoints2d
        selected_keypoints2d_list = []
        all_pred_kps2d = np.zeros((n_view, n_frame, 1, n_kps, 3))
        mview_person_id = [[] for _ in range(n_view)]
        for view_idx in range(n_view):
            pred_kps2d = np.zeros((n_frame, 1, n_kps, 3))
            max_n_kps2d = 1
            for frame_idx, sframe_keypoints2d in enumerate(selected_keypoints2d):
                kps2d = sframe_keypoints2d[view_idx].get_keypoints()
                n_kps2d = sframe_keypoints2d[view_idx].get_person_number()
                if n_kps2d > max_n_kps2d:
                    pred_kps2d = np.concatenate((pred_kps2d, np.zeros(
                        (n_frame, (n_kps2d - max_n_kps2d), n_kps, 3))), axis=1)
                    max_n_kps2d = n_kps2d
                pred_kps2d[frame_idx, :n_kps2d] = kps2d[0]
                mview_person_id[view_idx].append(np.array(range(n_kps2d)))
            selected_keypoints2d_list.append(
                Keypoints(
                    kps=pred_kps2d,
                    mask=pred_kps2d[..., -1] > 0,
                    convention=self.pred_kps3d_convention))
            n_person_all = all_pred_kps2d.shape[2]
            n_person_cur = pred_kps2d.shape[1]
            if n_person_all < n_person_cur:
                all_pred_kps2d = np.concatenate((all_pred_kps2d, np.zeros(
                    (n_view, n_frame, (n_person_cur - n_person_all), n_kps, 3))), axis=2)
            all_pred_kps2d[view_idx, :, 0:n_person_cur, ...] = pred_kps2d

        def solve_camera_pose(kps3d, all_kps2d, camera_params):
            for view_idx in range(all_kps2d.shape[0]):
                v_kps3d = kps3d[:, :, 5:]
                v_kps2d = all_kps2d[view_idx][:, :, 5:]
                valids = np.logical_and(v_kps3d[...,3] > 0, v_kps2d[...,2] > 0)
                c_kps3d = v_kps3d[np.where(valids)][...,0:3].astype(np.float32)
                c_kps2d = v_kps2d[np.where(valids)][...,0:2].astype(np.float32)
                camera_param:FisheyeCameraParameter = camera_params[view_idx]
                camera_matrix = np.array(camera_param.get_intrinsic())
                distort_coef = np.array(camera_param.get_dist_coeff()).reshape((-1,1))
                ret, rvecs, tvecs = cv2.solvePnP(c_kps3d, c_kps2d, camera_matrix, distort_coef)
                rotM = cv2.Rodrigues(rvecs)[0]
                print(ret, rvecs, tvecs, rotM)
        #solve_camera_pose(pred_kps3d, all_pred_kps2d, cam_param)

        self.logger.info('Finished estimating human 2d&3d keypoints.')
        self.logger.info('Optimizing human 3d keypoints...')

        # Optimizing keypoints3d
        if self.optimize_kps3d:
            optim_kwargs = dict(
                mview_kps2d=all_pred_kps2d,
                mview_kps2d_mask=all_pred_kps2d[..., 2:3] > 0,
                keypoints2d=selected_keypoints2d_list,
                mview_person_id=mview_person_id,
                matched_list=association_results,
                cam_params=cam_param)
            pred_keypoints3d = self.optimize_keypoints3d(pred_keypoints3d, **optim_kwargs)
        
        self.logger.info('Finished optimizing human 3d keypoints.')
        self.logger.info('Estimating human SMPL model...')

        # Fitting SMPL model
        if self.output_smpl:
            smpl_data_list = self.estimate_smpl(keypoints3d=pred_keypoints3d)
        else:
            smpl_data_list = None

        self.logger.info('Finished estimating human SMPL model.')
        self.logger.info('Finished processing video human pose.')

        return selected_keypoints2d_list, pred_keypoints3d, smpl_data_list

    def estimate_keypoints2d(
        self, img_arr: Union[None, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Keypoints]]:
        """Estimate bbox2d and keypoints2d.

        Args:
            img_arr (Union[None, np.ndarray], optional):
                A multi-view image array, in shape
                [n_view, n_frame, h, w, c]. Defaults to None.
        Returns:
            Tuple[List[np.ndarray], List[Keypoints]]:
                A list of bbox2d, and a list of keypoints2d instances.
        """

        mview_bbox2d_list = []
        mview_keypoints2d_list = []
        for view_index in range(img_arr.shape[0]):
            bbox2d_list = self.algo_bbox_detector.infer_array(
                image_array=img_arr[view_index], disable_tqdm=True,
                multi_person=self.multi_person)
            kps2d_list, _, bbox2d_list = self.algo_kps2d_estimator.infer_array(
                image_array=img_arr[view_index],
                bbox_list=bbox2d_list, disable_tqdm=True)
            keypoints2d = self.algo_kps2d_estimator.get_keypoints_from_result(
                kps2d_list)
            mview_bbox2d_list.append(bbox2d_list)
            mview_keypoints2d_list.append(keypoints2d)
            #print('keypoints2d:', keypoints2d)
            #visualize_keypoints2d(keypoints2d, 'temps/', only_show=True,
            #    background_arr=img_arr[view_index])

        return mview_bbox2d_list, mview_keypoints2d_list

    def select_camera(self, points: np.ndarray,
                      points_mask: np.ndarray) -> List[int]:
        """Use cam_pre_selector to filter bad points, use reprojection error of
        the good points to select good cameras.

        Args:
            cam_param (List[FisheyeCameraParameter]):
                A list of FisheyeCameraParameter instances.
            points (np.ndarray):
                Multiview points2d, in shape [n_view, n_kps, 3].
                Point scores at the last dim.
            points_mask (np.ndarray):
                Multiview points2d mask,
                in shape [n_view, n_kps, 1].

        Returns:
            List[int]: A list of camera indexes.
        """
        if self.algo_cam_selector is not None:
            self.logger.info('Selecting cameras.')
            if self.algo_cam_pre_selector is not None:
                self.logger.info('Using pre-selector for camera selection.')
                pre_mask = self.algo_cam_pre_selector.get_selection_mask(
                    points=points, init_points_mask=points_mask)
            else:
                pre_mask = points_mask.copy()
            self.algo_cam_selector.triangulator = self.algo_triangulator
            selected_camera_indexes = self.algo_cam_selector.get_camera_indexes(
                points=points, init_points_mask=pre_mask)
            self.logger.info(f'Selected cameras: {selected_camera_indexes}')
        else:
            self.logger.warning(
                'The estimator api instance has no cam_selector,' +
                ' all the cameras will be returned.')
            selected_camera_indexes = [idx for idx in range(len(points))]

        return selected_camera_indexes

    def estimate_keypoints3d(self, bbox2d_list, keypoints2d_list, mview_batch_arr, max_identity,
            pred_kps3d, selected_keypoints2d:list, association_results:list) -> int:
        n_frame = pred_kps3d.shape[0]
        n_kps = pred_kps3d.shape[2]
        frame_idx = len(selected_keypoints2d)
        n_view = len(keypoints2d_list)
        sframe_bbox2d_list = []
        sframe_keypoints2d_list = []
        sframe_association_results = []
        sframe_person_counts = []
        identities = []

        for view_idx in range(n_view):
            sview_kps2d_idx = []
            for idx, bbox2d in enumerate(bbox2d_list[view_idx][0]):
                if bbox2d[-1] > self.bbox_thr:
                    sview_kps2d_idx.append(idx)
            sview_kps2d_idx = np.array(sview_kps2d_idx)
            if len(sview_kps2d_idx) > 0:
                sframe_bbox2d_list.append(
                    torch.tensor(bbox2d_list[view_idx][0])[sview_kps2d_idx])
                mframe_keypoints2d = keypoints2d_list[view_idx]
                keypoints2d = Keypoints(
                    kps=mframe_keypoints2d.get_keypoints()[0:1, sview_kps2d_idx],
                    mask=mframe_keypoints2d.get_mask()[0:1, sview_kps2d_idx],
                    convention=mframe_keypoints2d.get_convention(),
                    logger=self.logger)
                if keypoints2d.get_convention() != self.pred_kps3d_convention:
                    keypoints2d = convert_keypoints(
                        keypoints=keypoints2d,
                        dst=self.pred_kps3d_convention,
                        approximate=True)
                sframe_keypoints2d_list.append(keypoints2d)
            else:
                sframe_bbox2d_list.append(torch.tensor([]))
                sframe_keypoints2d_list.append(
                    Keypoints(
                        kps=np.zeros((1, 1, n_kps, 3)),
                        mask=np.zeros((1, 1, n_kps)),
                        convention=self.pred_kps3d_convention))
            sframe_person_counts.append(len(sview_kps2d_idx))
        #sframe_person_counts[-1] = 0

        # Establish cross-frame and cross-person associations
        if self.multi_person:
            sframe_association_results, predict_keypoints3d, identities = \
                self.algo_associator.associate_frame(
                    # Dimension definition varies between
                    # cv2 images and tensor images.
                    mview_img_arr=mview_batch_arr[:, 0].transpose(0, 3, 1, 2),
                    mview_bbox2d=sframe_bbox2d_list,
                    mview_keypoints2d=sframe_keypoints2d_list,
                    #affinity_type='geometry_mean'
                    affinity_type='ReID only'
                )
        else:
            association_result = []
            valid_views = 0
            for idx, count in enumerate(sframe_person_counts):
                if count > 0:
                    association_result.append(0)
                    valid_views += 1
                else:
                    association_result.append(np.nan)
            if valid_views >= 2:
                sframe_association_results.append(association_result)
                identities.append(0)

        for p_idx in range(len(sframe_association_results)):
            # Triangulation, one associated person per time
            identity = identities[p_idx]
            associate_idxs = sframe_association_results[p_idx]
            tri_kps2d = np.zeros((n_view, n_kps, 3))
            tri_mask = np.zeros((n_view, n_kps, 1))
            for view_idx in range(n_view):
                kps2d_idx = associate_idxs[view_idx]
                if not np.isnan(kps2d_idx):
                    tri_kps2d[view_idx] = sframe_keypoints2d_list[
                        view_idx].get_keypoints()[0, int(kps2d_idx)]
                    tri_mask[view_idx, :, 0] = sframe_keypoints2d_list[
                        view_idx].get_mask()[0, int(kps2d_idx)]
            cam_indexes = self.select_camera(tri_kps2d, tri_mask)
            for i_view in range(0, n_view):
                if i_view not in cam_indexes:
                    tri_mask[i_view, ...] = 0
            if self.algo_point_selectors is not None:
                for selector in self.algo_point_selectors:
                    tri_mask = selector.get_selection_mask(
                        points=tri_kps2d, init_points_mask=tri_mask)
            kps3d = self.algo_triangulator.triangulate(tri_kps2d, tri_mask)
            if identity > max_identity:
                n_identity = identity - max_identity
                pred_kps3d = np.concatenate(
                    (pred_kps3d, np.zeros((n_frame, n_identity, n_kps, 4))), axis=1)
                max_identity = identity
            nan_mask = np.isnan(kps3d).any(axis=1)
            kps3d_mask = np.ones_like(kps3d[:, 0:1])
            kps3d_mask[nan_mask] = 0
            kps3d[nan_mask] = 0
            pred_kps3d[frame_idx, identity] = np.concatenate((kps3d, kps3d_mask), axis=-1)

        for identity in sorted(identities):
            index = identities.index(identity)
            association_results[frame_idx].append(sframe_association_results[index])
            print('identity: ', identity, sframe_association_results[index])
        selected_keypoints2d.append(sframe_keypoints2d_list)
        
        return max_identity, pred_kps3d

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                             **optim_kwargs) -> Keypoints:
        """Optimize keypoints3d.

        Args:
            keypoints3d (Keypoints): A keypoints3d Keypoints instance
        Returns:
            Keypoints: The optimized keypoints3d.
        """
        if self.algo_kps3d_optimizers is not None:
            for optimizer in self.algo_kps3d_optimizers:
                if hasattr(optimizer, 'triangulator'):
                    optimizer.triangulator = self.algo_triangulator
                keypoints3d = optimizer.optimize_keypoints3d(
                    keypoints3d, **optim_kwargs)
        return keypoints3d

    def estimate_smpl(self,
                      keypoints3d: Keypoints,
                      init_smpl_data: Union[None, SMPLData] = None,
                      return_joints: bool = False,
                      return_verts: bool = False) -> SMPLData:
        """Estimate smpl parameters according to keypoints3d.

        Args:
            keypoints3d (Keypoints):
                A keypoints3d Keypoints instance, with only one person
                inside. This method will take the person at
                keypoints3d.get_keypoints()[:, 0, ...] to run smplify.
            init_smpl_dict (dict, optional):
                A dict of init parameters. init_dict.keys() is a
                sub-set of self.__class__.OPTIM_PARAM.
                Defaults to an empty dict.
            return_joints (bool, optional):
                Whether to return joints. Defaults to False.
            return_verts (bool, optional):
                Whether to return vertices. Defaults to False.

        Returns:
            SMPLData:
                Smpl data of the person.
        """
        self.logger.info('Estimating SMPL.')
        working_convention = self.algo_smplify.body_model.keypoint_convention

        n_frame = keypoints3d.get_frame_number()
        n_person = keypoints3d.get_person_number()
        keypoints3d = keypoints3d.to_tensor(device=self.algo_smplify.device)
        person_mask = keypoints3d.get_mask()
        person_mask = torch.sum(person_mask, dim=2) > 0

        keypoints3d = convert_keypoints(keypoints=keypoints3d, dst=working_convention)
        kps3d_tensor = keypoints3d.get_keypoints()[:, :, :, :3].float()
        kps3d_conf = keypoints3d.get_mask()[:, :, ...]

        smpl_data_list = []
        for person in range(n_person):
            if person_mask[:, person].sum() == 0:
                continue

            # load init smpl data
            if init_smpl_data is not None:
                init_smpl_dict = init_smpl_data.to_tensor_dict(device=self.smplify.device)
            else:
                init_smpl_dict = {}

            global_orient = torch.zeros((n_frame, 3)).to(self.algo_smplify.device)
            transl = torch.full((n_frame, 3), 1000.0).to(self.algo_smplify.device)
            body_pose = torch.zeros((n_frame, 69)).to(self.algo_smplify.device)
            betas = torch.zeros((n_frame, 10)).to(self.algo_smplify.device)
            s_kps3d_tensor = kps3d_tensor[:, person][person_mask[:, person]]
            s_kps3d_conf = kps3d_conf[:, person][person_mask[:, person]]

            # build and run
            kp3d_mse_input = build_handler(
                dict(
                    type='Keypoint3dMSEInput',
                    keypoints3d=s_kps3d_tensor,
                    keypoints3d_conf=s_kps3d_conf,
                    keypoints3d_convention=working_convention,
                    handler_key='keypoints3d_mse'))
            kp3d_llen_input = build_handler(
                dict(
                    type='Keypoint3dLimbLenInput',
                    keypoints3d=s_kps3d_tensor,
                    keypoints3d_conf=s_kps3d_conf,
                    keypoints3d_convention=working_convention,
                    handler_key='keypoints3d_limb_len'))

            registrant_output = self.algo_smplify(
                input_list=[kp3d_mse_input, kp3d_llen_input],
                init_param_dict=init_smpl_dict,
                return_joints=return_joints,
                return_verts=return_verts)

            if self.smpl_data_type == 'smplx':
                smpl_data = SMPLXData()
            else: #elif self.smpl_data_type == 'smpl':
                smpl_data = SMPLData()

            global_orient[person_mask[:, person]] = registrant_output['global_orient']
            transl[person_mask[:, person]] = registrant_output['transl']
            body_pose[person_mask[:, person]] = registrant_output['body_pose']
            betas[person_mask[:, person]] = registrant_output['betas']
            output = {}
            output['global_orient'] = global_orient
            output['transl'] = transl
            output['body_pose'] = body_pose
            output['betas'] = betas
            smpl_data.from_param_dict(output)
            if return_joints:
                smpl_data['joints'] = registrant_output['joints']
            if return_verts:
                smpl_data['vertices'] = registrant_output['vertices']
            smpl_data_list.append(smpl_data)

        return smpl_data_list

