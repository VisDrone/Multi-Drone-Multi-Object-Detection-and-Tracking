# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys

import numpy as np
import torch
import tqdm
from torch.backends import cudnn


sys.path.append('./fast_reid_master/') 
from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from demo.new_predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

# import some modules added in project
# for example, add partial reid like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        # default = "./fast_reid_master/logs/mdmtreid/sbs_R50-ibn-pre/config.yaml",
        # default = "/root/autodl-tmp/mdmtreid/sbs_R50-ibn-preveri/config.yaml",
        # default = "/root/autodl-tmp/mdmtreid/sbs_R50-ibn-pre/config.yaml",
        default="F:/A_Master_Menu/TMM/github/mmtracking-master/fast_reid_master/configs/MDMTREID/config.yaml",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        help="a test dataset name for visualizing ranking list.",
        default='MDMTREID'
    )
    parser.add_argument(
        "--output",
        default="./fast_reid_master/demo/vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance",
        default=True
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="descending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="descending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def init_reid_model():
    args = get_parser().parse_args()
    # args.config_file = "./fast_reid_master/configs/MDMTREID/sbs_R50-ibn-preveri.yml"  # config路径
    args.dataset_name = 'MDMTREID'  # 数据集名字
    # args.vis_label = False  # 是否显示正确label结果
    args.rank_sort = 'descending'  # 从大到小显示关联结果
    args.label_sort = 'descending'  # 从大到小显示关联结果

    cfg = setup_cfg(args)
    # cfg["MODEL"]["WEIGHTS"] = './fast_reid_master/logs/mdmtreid/sbs_R50-ibn-pre/model_final.pth'
    # cfg["MODEL"]["WEIGHTS"] = '/root/autodl-tmp/mdmtreid/sbs_R50-ibn-preveri/model_final.pth'
    cfg["MODEL"]["WEIGHTS"] = './fast_reid_master/logs/mdmtreid/sbs_R50-ibn-pre/model_final.pth'
    
    # print(test_loader, num_query)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    return demo, args

def faster_reid_main(cropped_images, ids, camids_, num_query, demo, args):
    # 使用读写图像的方式加载进dataloader:
    # test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name)
    # 直接将追踪结果从原图中crop出来，不进行文件存储与dataloader的构建
    cropped_images = cropped_images
    
    
    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []
    for (feat, pid, camid) in tqdm.tqdm(demo.run_on_croped_images(cropped_images, ids, camids_), total=len(cropped_images)):
        feats.append(feat)
        pids.extend([pid])
        camids.extend([camid])

    
    print(len(feats))
    feats = torch.cat(feats, dim=0)  # ?????????????????????????????
    # feats = torch.tensor(feats)
    print(feats.shape)
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])

    # compute cosine distance
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()
    # print("distmap:  ********\n", distmat.shape)
    # print(distmat)

    logger.info("Computing APs for all query images ...")
    cmc, all_ap, all_inp = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Finish computing APs for all query images!")

    # visualizer = Visualizer(test_loader.dataset)
    visualizer = Visualizer(ids)
    visualizer.get_model_output(all_ap, distmat, q_pids, g_pids, q_camids, g_camids)

    # logger.info("Start saving ROC curve ...")
    # fpr, tpr, pos, neg = visualizer.vis_roc_curve(args.output)
    # visualizer.save_roc_info(args.output, fpr, tpr, pos, neg)
    # logger.info("Finish saving ROC curve!")

    logger.info("Saving rank list result ...")
    idc_dic, dist_list = visualizer.vis_rank_list_mdmt(distmat, args.output, args.vis_label, args.num_vis,
                                             args.rank_sort, args.label_sort, args.max_rank)
    logger.info("Finish saving rank list results!")
    print("idc_dic = ",idc_dic)
    print(dist_list)
    print("mean(dist_list)", np.mean(dist_list))
    return idc_dic, dist_list



if __name__ == '__main__':
    faster_reid_main()
