# encoding: utf-8
"""
@ author: Guanlin Chen
@ version: 2022/07/03 12:04:54
@ description:   mdmtreid dataloader
"""

import sys
import os
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY



TRAIN_DIR_KEY = 'train_dir'
VAL_DIR_KEY = 'val_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {

    # 'mdmtreid': {
    #     TRAIN_DIR_KEY: 'train_imgs',
    #     VAL_DIR_KEY:'val_imgs',
    #     TEST_DIR_KEY: 'test_imgs',
    # },
    'reid': {
        TRAIN_DIR_KEY: 'test_imgs',
        VAL_DIR_KEY:'test_imgs',
        TEST_DIR_KEY: 'test_imgs',
    },

}


@DATASET_REGISTRY.register()
class MDMTREID(ImageDataset):

    dataset_url = None
    dataset_name = 'mdmt'

    def __init__(self, root='/root/mmtracking-master/fast_reid_master/datasets/', **kwargs):
        # self.dataset_dir = root
        self.dataset_dir = '/root/mmtracking-master/fast_reid_master/datasets/'

        has_main_dir = False
        for main_dir in VERSION_DICT:                   # 获得训练和测试的directory
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                val_dir = VERSION_DICT[main_dir][VAL_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                has_main_dir = True
                break
        assert has_main_dir, 'Dataset folder not found'

        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.val_dir = osp.join(self.dataset_dir, main_dir, val_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        # self.list_train_path = osp.join(self.dataset_dir, main_dir, 'train_small.txt')
        # self.list_val_path = osp.join(self.dataset_dir, main_dir, 'val_small.txt')
        self.list_train_path = osp.join(self.dataset_dir, main_dir, 'query_1_small.txt')
        self.list_val_path = osp.join(self.dataset_dir, main_dir, 'gallery_2_small.txt')
        self.list_query_path = osp.join(self.dataset_dir, main_dir, 'query_1_small.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, main_dir, 'gallery_2_small.txt')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.val_dir,
            self.test_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.val_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path, is_train=False)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path, is_train=False)

        num_train_pids = self.get_num_pids(train)
        query_tmp = []
        for img_path, pid, camid in query:
            query_tmp.append((img_path, pid+num_train_pids, camid))
        del query
        query = query_tmp

        gallery_temp = []
        for img_path, pid, camid in gallery:
            gallery_temp.append((img_path, pid+num_train_pids, camid))
        del gallery
        gallery = gallery_temp

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        if 'combineall' in kwargs and kwargs['combineall']:
            train += val
        super(MDMTREID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path, is_train=True):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            # camid = int((list(img_path.split('/')[1]))[3]) 
            camid_pre = img_path.split('/')[1]
            camid = int((list(camid_pre.split('-')[1]))[0]) 
            img_path = osp.join(dir_path, img_path)
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
