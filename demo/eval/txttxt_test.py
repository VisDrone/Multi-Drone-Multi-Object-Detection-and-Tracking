# _*_ coding: utf-8 _*_
# @Time    :2022/7/14 21:18
# @Author  :LiuZhihao
# @File    :txttxt_test.py

import motmetrics as mm
import numpy as np
import os
from argparse import ArgumentParser

print("hello")

parser == ArgumentParser()
parser.add_argument('--test_file_dir', default="./demo/txt/MOT/Firstframe_initialized_faster_rcnn_r50_fpn_carafe_1x_full_mdmt/", help="test file directory")
args = parser.parse_args()

#评价指标
metrics = list(mm.metrics.motchallenge_metrics)
#导入gt和ts文件
gt_files_dir = "./demo/txt/gt_true"
ts_files_dir = args.test_file_dir
# gt=mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
# ts=mm.io.loadtxt(ts_file, fmt="mot15-2D")
sequences1 = ["26-1", "31-1", "34-1", "48-1", "52-1", "55-1", "56-1", "57-1", "59-1", "61-1", "62-1", "68-1", "71-1", "73-1"]
sequences2 = ["26-2", "31-2", "34-2", "48-2", "52-2", "55-2", "56-2", "57-2", "59-2", "61-2", "62-2", "68-2", "71-2", "73-2"]
frames = [300, 360, 300, 700, 360, 151, 400, 490, 700, 270, 700, 310, 250, 592]



idf11_sum = 0
mota_sum = 0
nn = 0
idf11_sum = []
mota_sum = []
lsdir = os.listdir(ts_files_dir)
lsdir.sort()
for ts_file_dir in lsdir:
    if "-1" in ts_file_dir:
        name = ts_file_dir.split(".")[0]
        ind_ = sequences1.index(name)
        frame = frames[ind_]
        # continue
    elif "-2" in ts_file_dir:
        name = ts_file_dir.split(".")[0]
        ind_ = sequences2.index(name)
        frame = frames[ind_]
        # continue
    else:
        continue
    gt_file = os.path.join(gt_files_dir, ts_file_dir)
    ts_file = os.path.join(ts_files_dir, ts_file_dir)
    gt=mm.io.loadtxt(gt_file,  min_confidence=1)
    ts=mm.io.loadtxt(ts_file)
    name=os.path.splitext(os.path.basename(ts_file))[0]
    #计算单个acc
    acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metrics, name=name)
    # print(summary["idf1"])
    idf11 = float(summary["idf1"])
    mota = float(summary["mota"])
    idf11_sum.append(idf11)
    mota_sum.append(mota)
    nn += 1
    print(idf11)
    print(mota)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
idf11_averagee = sum(idf11_sum)/nn
mota_averagee = sum(mota_sum)/nn
Aidf1 = 0
Bidf1 = 0
for i in range(len(idf11_sum)):
    if i%2 == 0:
        Aidf1 += idf11_sum[i]
    else:
        Bidf1 += idf11_sum[i]
    
Aidf1 = 2*Aidf1/nn
Bidf1 = 2*Bidf1/nn

Amota = 0
Bmota = 0
for i in range(len(mota_sum)):
    if i%2 == 0:
        Amota += mota_sum[i]
    else:
        Bmota += mota_sum[i]
    
Amota = 2*Amota/nn
Bmota = 2*Bmota/nn


print(idf11_averagee)
print(mota_averagee)
print("A idf1:", Aidf1)
print("B idf1:", Bidf1)
print("A mota:", Amota)
print("B mota:", Bmota)


# 计算每个序列增加了多少目标
# import os
# ts_files_dir="./demo/txt/MDMT/pure_supplement_counting/NMS-bytetrack_autoassign_full_mdmt-private-half/"
# files = os.listdir(ts_files_dir)
# files.sort()
# for file in files:
#     if '.ipynb_checkpoints' == file:
#         continue
#     print(file)
#     with open(os.path.join(ts_files_dir, file)) as file:
#         print(len(file.readlines()))



#
#
# # py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# # https://github.com/cheind/py-motmetrics/
#
# # MIT License
# # Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# # See LICENSE file for terms.
#
# """Compute metrics for trackers using MOTChallenge ground-truth data."""
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import argparse
# from collections import OrderedDict
# import glob
# import logging
# import os
# from pathlib import Path
#
# import motmetrics as mm
#
#
# def parse_args():
#     """Defines and parses command-line arguments."""
#     parser = argparse.ArgumentParser(description="""
# Compute metrics for trackers using MOTChallenge ground-truth data.
#
# Files
# -----
# All file content, ground truth and test files, have to comply with the
# format described in
#
# Milan, Anton, et al.
# "Mot16: A benchmark for multi-object tracking."
# arXiv preprint arXiv:1603.00831 (2016).
# https://motchallenge.net/
#
# Structure
# ---------
#
# Layout for ground truth data
#     <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
#     <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
#     ...
#
# Layout for test data
#     <TEST_ROOT>/<SEQUENCE_1>.txt
#     <TEST_ROOT>/<SEQUENCE_2>.txt
#     ...
#
# Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
# string.""", formatter_class=argparse.RawTextHelpFormatter)
#
#     parser.add_argument('-groundtruths', default="/home/linkdata/data/tmm/mmlap/mmtracking-master/data/FULL_MDMT/test/", type=str, help='Directory containing ground truth files.')
#     parser.add_argument('-tests', default="/home/linkdata/data/tmm/eval/txttxt/raw_mot/", type=str, help='Directory containing tracker result files')
#     parser.add_argument('--loglevel', type=str, help='Log level', default='info')
#     parser.add_argument('--fmt', type=str, help='Data format', default='mot16')
#     parser.add_argument('--solver', type=str, help='LAP solver to use for matching between frames.')
#     parser.add_argument('--id_solver', type=str, help='LAP solver to use for ID metrics. Defaults to --solver.')
#     parser.add_argument('--exclude_id', dest='exclude_id', default=False, action='store_true',
#                         help='Disable ID metrics')
#     return parser.parse_args()
#
#
# def compare_dataframes(gts, ts):
#     """Builds accumulator for each sequence."""
#     accs = []
#     names = []
#     for k, tsacc in ts.items():
#         if k in gts:
#             logging.info('Comparing %s...', k)
#             accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, distth=0.5))  #, 'iou'
#             names.append(k)
#         else:
#             logging.warning('No ground truth for %s, skipping.', k)
#
#     return accs, names
#
#
# def main():
#     # pylint: disable=missing-function-docstring
#     args = parse_args()
#
#     loglevel = getattr(logging, args.loglevel.upper(), None)
#     if not isinstance(loglevel, int):
#         raise ValueError('Invalid log level: {} '.format(args.loglevel))
#     logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')
#
#     if args.solver:
#         mm.lap.default_solver = args.solver
#
#     gtfiles = glob.glob(os.path.join(args.groundtruths, '*/gt/gt.txt'))
#     tsfiles = [f for f in glob.glob(os.path.join(args.tests, '*.txt')) if not os.path.basename(f).startswith('eval')]
#
#     logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
#     logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
#     logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
#     logging.info('Loading files.')
#
#     gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for f in gtfiles])
#     ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.fmt)) for f in tsfiles])
#
#     mh = mm.metrics.create()
#     accs, names = compare_dataframes(gt, ts)
#
#     metrics = list(mm.metrics.motchallenge_metrics)
#     if args.exclude_id:
#         metrics = [x for x in metrics if not x.startswith('id')]
#
#     logging.info('Running metrics')
#
#     if args.id_solver:
#         mm.lap.default_solver = args.id_solver
#     summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
#     print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
#     logging.info('Completed')
#
#
# if __name__ == '__main__':
#     main()