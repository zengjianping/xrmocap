import mmengine
import os
import pytest
import shutil
import torch
from xrprimer.utils.path_utils import Existence, check_path_existence

from xrmocap.data.data_converter.builder import build_data_converter

input_dir = 'tests/data/data/test_data_converter'
output_dir = 'tests/data/output/data/test_data_converter'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    dataset_names = ['Campus_unittest', 'Shelf_unittest', 'panoptic_unittest']
    for name in dataset_names:
        src_path = os.path.join(input_dir, name)
        dst_path = os.path.join(output_dir, name)
        shutil.copytree(src_path, dst_path)


def test_convert_campus():
    converter_config = dict(
        mmengine.Config.fromfile('configs/modules/data/data_converter/' +
                             'campus_data_converter_unittest.py'))
    data_root = os.path.join(output_dir, 'Campus_unittest')
    meta_path = os.path.join(data_root, 'test_convert_campus')
    converter_config['meta_path'] = meta_path
    converter_config['data_root'] = data_root
    converter_config['bbox_detector'] = None
    converter_config['kps2d_estimator'] = None
    converter = build_data_converter(converter_config)
    converter.run(overwrite=True)
    assert check_path_existence(
        meta_path, 'dir') == \
        Existence.DirectoryExistNotEmpty


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU device has been found.')
def test_convert_campus_perception2d():
    converter_config = dict(
        mmengine.Config.fromfile('configs/modules/data/data_converter/' +
                             'campus_data_converter_unittest.py'))
    data_root = os.path.join(output_dir, 'Campus_unittest')
    meta_path = os.path.join(data_root, 'test_convert_campus_perception2d')
    converter_config['meta_path'] = meta_path
    converter_config['data_root'] = data_root
    converter = build_data_converter(converter_config)
    converter.run(overwrite=True)
    assert check_path_existence(
        meta_path, 'dir') == \
        Existence.DirectoryExistNotEmpty


def test_convert_shelf():
    converter_config = dict(
        mmengine.Config.fromfile('configs/modules/data/data_converter/' +
                             'shelf_data_converter_unittest.py'))
    data_root = os.path.join(output_dir, 'Shelf_unittest')
    meta_path = os.path.join(data_root, 'test_convert_shelf')
    converter_config['meta_path'] = meta_path
    converter_config['data_root'] = data_root
    converter_config['bbox_detector'] = None
    converter_config['kps2d_estimator'] = None
    converter = build_data_converter(converter_config)
    converter.run(overwrite=True)
    assert check_path_existence(
        meta_path, 'dir') == \
        Existence.DirectoryExistNotEmpty


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU device has been found.')
def test_convert_shelf_perception2d():
    converter_config = dict(
        mmengine.Config.fromfile('configs/modules/data/data_converter/' +
                             'shelf_data_converter_unittest.py'))
    data_root = os.path.join(output_dir, 'Shelf_unittest')
    meta_path = os.path.join(data_root, 'test_convert_shelf_perception2d')
    converter_config['meta_path'] = meta_path
    converter_config['data_root'] = data_root
    converter = build_data_converter(converter_config)
    converter.run(overwrite=True)
    assert check_path_existence(
        meta_path, 'dir') == \
        Existence.DirectoryExistNotEmpty


def test_convert_panoptic():
    converter_config = dict(
        mmengine.Config.fromfile('configs/modules/data/data_converter/' +
                             'panoptic_data_converter_unittest.py'))
    data_root = os.path.join(output_dir, 'panoptic_unittest')
    meta_path = os.path.join(data_root, 'test_convert_panoptic')
    converter_config['meta_path'] = meta_path
    converter_config['data_root'] = data_root
    converter_config['bbox_detector'] = None
    converter_config['kps2d_estimator'] = None
    converter = build_data_converter(converter_config)
    converter.run(overwrite=True)
    assert check_path_existence(
        meta_path, 'dir') == \
        Existence.DirectoryExistNotEmpty


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU device has been found.')
def test_convert_panoptic_perception2d():
    converter_config = dict(
        mmengine.Config.fromfile('configs/modules/data/data_converter/' +
                             'panoptic_data_converter_unittest.py'))
    data_root = os.path.join(output_dir, 'panoptic_unittest')
    meta_path = os.path.join(data_root, 'test_convert_panoptic_perception2d')
    converter_config['meta_path'] = meta_path
    converter_config['data_root'] = data_root
    converter = build_data_converter(converter_config)
    converter.run(overwrite=True)
    assert check_path_existence(
        meta_path, 'dir') == \
        Existence.DirectoryExistNotEmpty
