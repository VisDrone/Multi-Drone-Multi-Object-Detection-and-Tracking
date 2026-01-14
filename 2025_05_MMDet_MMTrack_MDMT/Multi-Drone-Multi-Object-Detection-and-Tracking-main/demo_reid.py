# Copyright (c) OpenMMLab. All rights reserved.
from math import dist
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser
# from winreg import QueryInfoKey
from mmtrack.core.track.transforms import results2outs
import torch.nn.functional as F
import mmcv
import copy
from mmtrack.apis import inference_mot, init_model
# from mmtrack.apis import inference_mot, init_model,inference_reid_mdmt,inference_reid_mdmt_com
import torch
import os
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from mmcv.utils import mkdir_or_exist
import sys

# sys.path.append('./Person_reID_baseline_pytorch-master/')
# from demo_demo import demo
sys.path.append('./fast_reid_master/')

from fast_reid_master.demo.new_predictor import FeatureExtractionDemo
from fast_reid_master.demo.demo_fast_reid import faster_reid_main, init_reid_model
import xml.etree.ElementTree as ET
import time
import json


def read_xml_r(xml_file1, i):
    # 读取xml文件
    root1 = ET.parse(xml_file1).getroot()  # xml文件根
    track_all = root1.findall('track')
    # 初始化box及其对应的ids
    bboxes1 = []
    ids = []
    id = 0
    for track in track_all:
        id_label = int(track.attrib['id'])
        # label = int(category[track.attrib['label']])
        boxes = track.findall('box')
        shape = [1080, 1920]
        for box in boxes:
            # print(box.attrib['frame'])
            if int(box.attrib['frame']) == i:
                xtl = int(box.attrib['xtl'])
                ytl = int(box.attrib['ytl'])
                xbr = int(box.attrib['xbr'])
                ybr = int(box.attrib['ybr'])
                outside = int(box.attrib['outside'])
                occluded = int(box.attrib['occluded'])
                centx1 = int((xtl + xbr) / 2)
                centy1 = int((ytl + ybr) / 2)
                #
                if outside == 1 or xtl <= 10 and ytl <= 10 or xbr >= shape[1] - 0 and ytl <= 0 \
                        or xtl <= 10 and ybr >= shape[0] - 10 or xbr >= shape[1] - 10 and ybr >= shape[0] - 10:
                    break
                # cv2.rectangle(image1, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
                # cv2.putText(image1, str(id), (xtl, ybr), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
                # cv2.imshow("jj", image1)
                # cv2.waitKey(10)
                confidence = 0.99
                bboxes1.append([xtl, ytl, xbr, ybr, 1])
                ids.append(id_label)
                id += 1
                break
    bboxes1 = torch.tensor(bboxes1, dtype=torch.long)
    ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.zeros_like(ids)
    return bboxes1, ids, labels


def crop_tracks(img_,
                bboxes,
                score_thr=0.0,
                ):
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 5
    if isinstance(img_, str):
        img_ = mmcv.imread(img_)

    img_shape = img_.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    # inds = np.where(bboxes[:, -1] > score_thr)[0]  # ??????????????????????
    # print(len(bboxes))
    # bboxes = bboxes[inds]
    # print(len(bboxes))

    cropped_images = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox[:4].type(torch.int32)
        w1 = x2 - x1
        h1 = y2 - y1
        y11 = max(y1 - h1 / 2, 0)
        y22 = min(y2 + h1 / 2, 1080)
        x11 = max(0, x1 - w1 / 2)
        x22 = min(1920, x2 + w1 / 2)
        # data = data.type(torch.float32)

        # cut_color_img = img_[y1:y2, x1:x2]
        cut_color_img = img_[int(y11):int(y22), int(x11):int(x22)]
        cv2.imwrite("cut_image.jpg", cut_color_img)
        cropped_images.append(cut_color_img)

    return cropped_images


def id_updata(id_dic, result1, result2):
    ori_id1 = result1.get('track_ids_tensor')
    ori_id2 = result2.get('track_ids_tensor')
    bbox1 = result1.get('track_bboxes', None)
    tem_bbox10 = torch.from_numpy(bbox1[0])
    tem_bbox11 = torch.from_numpy(bbox1[1])
    tem_bbox12 = torch.from_numpy(bbox1[2])
    tem_bbox1 = torch.cat([tem_bbox10, tem_bbox11, tem_bbox12], 0)
    bbox2 = result2.get('track_bboxes', None)
    tem_bbox20 = torch.from_numpy(bbox2[0])
    tem_bbox21 = torch.from_numpy(bbox2[1])
    tem_bbox22 = torch.from_numpy(bbox2[2])
    tem_bbox2 = torch.cat([tem_bbox20, tem_bbox21, tem_bbox22], 0)
    print("id_dic = ", id_dic)
    print("result1.get('track_ids_tensor') = ", result1.get('track_ids_tensor'))
    print("result2.get('track_ids_tensor') = ", result2.get('track_ids_tensor'))
    com_num = max(len(ori_id1), len(ori_id1))
    tmp_ori_id2 = copy.copy(ori_id2)
    tmp_ori_id1 = copy.copy(ori_id1)
    for (i, id) in enumerate(id_dic):
        print("id1 = ", id[0])
        print("id2 = ", id[1])
        id1 = int(id[0])
        id2 = int(id[1])
        for j in range(0, len(ori_id2)):
            if (ori_id2[j] == int(id[1])):
                # ori_id2[j]=int(id[0])
                com_num = com_num + 1
                tmp_ori_id2[j] = com_num
                tmp_ori_id1[i] = com_num
                tem_bbox1[i][0] = com_num
                tem_bbox2[j][0] = com_num

    bbox1[0] = tem_bbox1[0:len(tem_bbox10)].numpy()
    bbox1[1] = tem_bbox1[len(tem_bbox10): len(tem_bbox10) + len(tem_bbox11)].numpy()
    bbox1[2] = tem_bbox1[len(tem_bbox10) + len(tem_bbox11):].numpy()
    bbox2[0] = tem_bbox2[0:len(tem_bbox20)].numpy()
    bbox2[1] = tem_bbox2[len(tem_bbox20): len(tem_bbox20) + len(tem_bbox21)].numpy()
    bbox2[2] = tem_bbox2[len(tem_bbox20) + len(tem_bbox21):].numpy()
    # print("ori_id1 = ",ori_id1)
    # print("ori_id2 = ",ori_id2)
    # print("bbox1 = ",bbox1)
    # print("bbox2 = ",bbox2)
    result1['track_bboxes'] = bbox1
    result2['track_bboxes'] = bbox2
    result1['track_ids_tensor'] = ori_id1
    result2['track_ids_tensor'] = ori_id2
    print("finally result1 = ", result1.get('track_bboxes'))
    print("finally result2 = ", result2.get('track_bboxes'))
    return result1, result2


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='./configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py',
                        help='config file')
    # parser.add_argument('--config', default='./configs/mot/bytetrack/one_carafe_bytetrack_full_mdmt.py',
    #                     help='config file')

    parser.add_argument('--input', default='F:/A_Master_Menu/_A_dataset/MCMOT-new/MDMT/1/',
                        help='input video file or folder')

    parser.add_argument('--xml_dir', default='F:/A_Master_Menu/_A_dataset/MCMOT-new/new_xml/',
                        help='input xml file of the groundtruth')

    parser.add_argument('--result_dir', default='./reid_json_resultfiles/refinedREID',
                        help='result_dir name, no "/" in the end')
    parser.add_argument('--method', default='autoassign',
                        help='the output directory name used in result_dir')

    parser.add_argument(
        '--output', default='./workdirs/map26.mp4', help='output video file (mp4 format) or folder')
    parser.add_argument(
        '--output2', default='./workdirs/map26B.mp4', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint',
                        help='checkpoint file, can be initialized in config files')  # , default="../workdirs/autoassign_epoch_60.pth"
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        # default=True,
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', default=10, help='FPS of the output video')
    args = parser.parse_args()
    assert args.output or args.show
    loopp = 0
    # load images
    track_bboxes_old = []
    track_bboxes2_old = []
    time_start_all = time.time()
    for dirrr in sorted(os.listdir(args.input)):
        if "-2" in dirrr:
            print("dirrr has -2")
            continue
        if "-1" not in dirrr and "-2" not in dirrr:
            continue
        # loopp += 1
        # if loopp < 4:
        #     continue
        # print(os.path.join(args.input+dirrr+"/"))
        sequence_dir = os.path.join(args.input + dirrr + "/")
        if osp.isdir(sequence_dir):
            imgs = sorted(
                filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                       # os.listdir(args.input)),
                       os.listdir(sequence_dir)),
                key=lambda x: int(x.split('.')[0]))
            IN_VIDEO = False
        else:
            # imgs = mmcv.VideoReader(args.input)
            imgs = mmcv.VideoReader(sequence_dir)
            IN_VIDEO = True
        # define output
        if args.output is not None:
            if args.output.endswith('.mp4'):
                OUT_VIDEO = True
                out_dir = tempfile.TemporaryDirectory()
                out_path = out_dir.name
                _out = args.output.rsplit(os.sep, 1)
                if len(_out) > 1:
                    os.makedirs(_out[0], exist_ok=True)
            else:
                OUT_VIDEO = False
                out_path = args.output
                os.makedirs(out_path, exist_ok=True)
        if args.output2 is not None:
            if args.output2.endswith('.mp4'):
                OUT_VIDEO = True
                out_dir2 = tempfile.TemporaryDirectory()
                out_path2 = out_dir2.name
                _out2 = args.output2.rsplit(os.sep, 1)
                if len(_out2) > 1:
                    os.makedirs(_out2[0], exist_ok=True)
            else:
                OUT_VIDEO = False
                out_path2 = args.output2
                os.makedirs(out_path2, exist_ok=True)

        fps = args.fps
        if args.show or OUT_VIDEO:
            if fps is None and IN_VIDEO:
                fps = imgs.fps
            if not fps:
                raise ValueError('Please set the FPS for the output video.')
            fps = int(fps)

        # build the model from a config file and a checkpoint file
        model = init_model(args.config, args.checkpoint, device=args.device)
        model2 = init_model(args.config, args.checkpoint, device=args.device)
        demo, args_reid = init_reid_model()
        # reid_model = init_model(args.reid_config, args.checkpoint, device=args.device)
        # reid_model = init_model(args.reid_config,  device=args.device)

        prog_bar = mmcv.ProgressBar(len(imgs))
        # test and show/save the images
        matched_ids = []
        A_max_id = 0
        B_max_id = 0

        result_dict = {}
        result_dict2 = {}
        time_start = time.time()
        # test and show/save the images
        for i, img in enumerate(imgs):
            flag = 0
            coID_confirme = []
            supplement_bbox = np.array([])
            supplement_bbox2 = np.array([])
            if isinstance(img, str):
                # img = osp.join(args.input, img)
                img = osp.join(sequence_dir, img)
                img2 = img.replace("/1/", "/2/")
                img2 = img2.replace("-1", "-2")
                # print(img2)
                image1 = cv2.imread(img)
                image2 = cv2.imread(img2)

            # for the first frame----offline
            if i == 0:
                # print("for the first frame----offline---given labels to update")
                sequence1 = img.split("/")[-2]
                xml_file1 = os.path.join(
                    "{}".format(args.xml_dir) + "{}".format(sequence1) + ".xml")
                # print(xml_file1)
                sequence2 = img2.split("/")[-2]
                xml_file2 = os.path.join(
                    "{}".format(args.xml_dir) + "{}".format(sequence2) + ".xml")
                # print(xml_file2)
                bboxes1, ids1, labels1 = read_xml_r(xml_file1, i)
                bboxes2, ids2, labels2 = read_xml_r(xml_file2, i)
                # print(bboxes1)

                # 第一帧做完之后不进行后续操作，后续操作从第二帧开始
                # continue

            # result1 = inference_mot(model, img, frame_id=i)
            # result2 = inference_mot(model2, img2, frame_id=i)

            # inference process
            max_id = max(A_max_id, B_max_id)
            result, max_id = inference_mot(model, img, frame_id=i, bboxes1=bboxes1, ids1=ids1, labels1=labels1,
                                           max_id=max_id)  # 如果在这边的参数名为后续用到的bboxes等，则会影响后面的参数，所以参数到底怎么传的需要继续学习
            # result = dict(det_bboxes=det_results['bbox_results'],
            #             track_bboxes=track_results['bbox_results'])
            det_bboxes = result['det_bboxes'][0]
            track_bboxes = result['track_bboxes'][0]
            result2, max_id = inference_mot(model2, img2, frame_id=i, bboxes1=bboxes2, ids1=ids2, labels1=labels2,
                                            max_id=max_id)
            det_bboxes2 = result2['det_bboxes'][0]
            track_bboxes2 = result2['track_bboxes'][0]
            print(track_bboxes.shape)
            print(track_bboxes2.shape)
            # ************************Re-ID************************************************

            # ***********************************************************************

            print("第 ", i, " 次结束 ")
            if args.output is not None:
                if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path, f'{i:06d}.jpg')
                    out_file2 = osp.join(out_path2, f'{i:06d}.jpg')
                else:
                    out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
                    out_file2 = osp.join(out_path2, img2.rsplit(os.sep, 1)[-1])
            else:
                out_file = None
                out_file2 = None
            # print("LLLLLLLLLLLLLLLLLLLLLLLLLL", result.keys(), result["track_bboxes"][0].shape[0])

            file_type = 'query' if result["track_bboxes"][0].shape[0] <= result2["track_bboxes"][0].shape[
                0] else 'gallery'
            # droneA
            A_bboxes_ = torch.tensor(track_bboxes[:, 1:6], dtype=torch.long)
            A_ids = torch.tensor(track_bboxes[:, 0], dtype=torch.long)
            A_labels_ = torch.zeros_like(torch.tensor(track_bboxes[:, 0]))
            A_cropped_images = crop_tracks(image1,
                                           A_bboxes_,
                                           score_thr=0.0,
                                           )
            A_camids = torch.ones_like(A_ids)
            # print("droneA camids: ", A_camids)

            # droneB
            B_bboxes_ = torch.tensor(track_bboxes2[:, 1:6], dtype=torch.long)
            B_ids = torch.tensor(track_bboxes2[:, 0], dtype=torch.long)
            B_labels_ = torch.zeros_like(torch.tensor(track_bboxes2[:, 0]))
            B_cropped_images = crop_tracks(image2,
                                           B_bboxes_,
                                           score_thr=0.0,
                                           )
            B_camids = torch.ones_like(B_ids) * 2
            # print("droneB camsid: ", B_camids)

            if file_type == "query":
                num_query = len(A_cropped_images)
                cropped_images = np.concatenate((A_cropped_images, B_cropped_images), axis=0)
                ids = np.concatenate((A_ids, B_ids), axis=0)
                camids = np.concatenate((A_camids, B_camids), axis=0)
            else:
                num_query = len(B_cropped_images)
                cropped_images = np.concatenate((B_cropped_images, A_cropped_images), axis=0)
                ids = np.concatenate((B_ids, A_ids), axis=0)
                camids = np.concatenate((B_camids, A_camids), axis=0)

            print(len(A_bboxes_), len(B_bboxes_), len(A_cropped_images), len(B_cropped_images), len(cropped_images),
                  len(ids), len(camids), num_query)
            print(camids)
            id_dic, dist_list = faster_reid_main(cropped_images, ids, camids, num_query, demo, args_reid)
            # time.sleep(10)
            # print(result["track_bboxes"])
            print("id_dic: ***********\n", id_dic)
            for num_i, _ in enumerate(dist_list):
                # if _ > -9.0:
                if _ > -16.0:
                    id_dic[num_i] = 0
            print(id_dic)

            print(track_bboxes[:, 0])
            print(track_bboxes2[:, 0])
            keep_matched = []
            for matched in id_dic:
                if matched == 0:
                    continue
                if matched[0] == matched[1]:
                    keep_matched.append(matched[0])
            for matched in id_dic:
                if matched == 0:
                    continue
                if matched[0] in keep_matched or matched[1] in keep_matched:
                    continue
                min_id = int(min(matched))
                if file_type == 'query':
                    # print("query--droneA")
                    # print("matched[0]", matched[0])
                    A_index = np.where(track_bboxes[:, 0] == int(matched[0]))[0].astype(int)
                    B_index = np.where(track_bboxes2[:, 0] == int(matched[1]))[0].astype(int)
                    # print(track_bboxes[A_index, 0])
                    # print(track_bboxes2[B_index, 0])
                    # print("******************")
                    track_bboxes[A_index, 0] = min_id
                    track_bboxes2[B_index, 0] = min_id

                    keep_matched.append(min_id)
                    # print(track_bboxes[A_index, 0])
                    # print(track_bboxes2[B_index, 0])
                    # print("******************")
                    # print("******************")
                else:
                    # print("query--droneB")
                    # print("matched[0]", matched[0])
                    A_index = np.where(track_bboxes[:, 0] == int(matched[1]))[0].astype(int)
                    # print(A_index)
                    B_index = np.where(track_bboxes2[:, 0] == int(matched[0]))[0].astype(int)
                    # print(track_bboxes[A_index, 0])
                    # print(track_bboxes2[B_index, 0])
                    # print("******************")
                    track_bboxes[A_index, 0] = min_id
                    track_bboxes2[B_index, 0] = min_id

                    keep_matched.append(min_id)

                    # print(track_bboxes[A_index, 0])
                    # print(track_bboxes2[B_index, 0])
                    # print("******************")
                    # print("******************")
            print(track_bboxes[:, 0])
            print(track_bboxes2[:, 0])

            A_max_id = max(A_max_id, max(track_bboxes[:, 0]))
            B_max_id = max(B_max_id, max(track_bboxes2[:, 0]))

            result['track_bboxes'][0] = track_bboxes
            result2['track_bboxes'][0] = track_bboxes2

            bboxes1 = torch.tensor(track_bboxes[:, 1:5], dtype=torch.long)
            ids1 = torch.tensor(track_bboxes[:, 0], dtype=torch.long)
            labels1 = torch.zeros_like(torch.tensor(track_bboxes[:, 0]))
            # labels1 = torch.tensor(track_bboxes[:, 0])

            bboxes2 = torch.tensor(track_bboxes2[:, 1:5], dtype=torch.long)
            ids2 = torch.tensor(track_bboxes2[:, 0], dtype=torch.long)
            labels2 = torch.zeros_like(torch.tensor(track_bboxes2[:, 0]))
            # new_result1 , new_result2 = id_updata(id_dic,result1= result1,result2 = result2)

            result_dict["frame={}".format(i)] = track_bboxes[:, 0:5].tolist()
            result_dict2["frame={}".format(i)] = track_bboxes2[:, 0:5].tolist()

            model.show_result(
                img,
                result,
                score_thr=args.score_thr,
                show=args.show,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file,
                backend=args.backend,
                new_id=[])
            model2.show_result(
                img2,
                result2,
                score_thr=args.score_thr,
                show=args.show,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file2,
                backend=args.backend,
                new_id=[])
            # tmp_query , gallery_label = demo()    

            # exit()

            prog_bar.update()

        time_end = time.time()
        method = args.method
        # method = "NMS-one_carafe_bytetrack_full_mdmt"
        json_dir = "{}/{}/".format(args.result_dir, method)

        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        with open("{0}/{1}.json".format(json_dir, sequence1), "w") as f:
            json.dump(result_dict, f, indent=4)
            print("输出文件A写入完成！")
        with open("{0}/{1}.json".format(json_dir, sequence2), "w") as f2:
            json.dump(result_dict2, f2, indent=4)
            print("输出文件B写入完成！")
        with open("{0}/time.txt".format(json_dir), "a") as f3:
            f3.write("{} time consume :{}\n".format(sequence1, time_end - time_start))
            print("输出文件time.txt写入完成！")

        if args.output and OUT_VIDEO:
            print(f'making the output video at {args.output} with a FPS of {fps}')
            # mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
            # mmcv.frames2video(out_path2, args.output2, fps=fps, fourcc='mp4v')
            # mmcv.frames2video(out_path3, args.output3, fps=fps, fourcc='mp4v')
            # mmcv.frames2video(out_path4, args.output4, fps=fps, fourcc='mp4v')
            # out_dir.cleanup()
            # out_dir2.cleanup()
            # out_dir3.cleanup()
            # out_dir4.cleanup()
    time_end_all = time.time()
    with open("{0}/time.txt".format(json_dir), "a") as f3:
        f3.write("ALL time consume :{}\n".format(time_end_all - time_start_all))


if __name__ == '__main__':
    main()
