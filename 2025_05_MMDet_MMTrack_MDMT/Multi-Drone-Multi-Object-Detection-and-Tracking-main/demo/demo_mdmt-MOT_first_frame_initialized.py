# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import copy
from mmtrack.apis import inference_mot, init_model # ,inference_reid_mdmt_com  # ,inference_reid_mdmt
import torch
import os 
import json
import numpy as np
import time
import xml.etree.ElementTree as ET
from threading import Thread
import cv2


def all_nms(dets, thresh):
    x1 = dets[:, 1]  # xmin
    y1 = dets[:, 2]  # ymin
    x2 = dets[:, 3]  # xmax
    y2 = dets[:, 4]  # ymax
    scores = dets[:, 5]  # confidence

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个boundingbox的面积
    order = scores.argsort()[::-1]  # boundingbox的置信度排序
    keep = np.array([])  # 用来保存最后留下来的boundingbox
    while order.size > 0:
        i = order[0]  # 置信度最高的boundingbox的index
        if keep.size == 0:
            keep = np.array([dets[i]])
        else:
            keep = np.append(keep, [dets[i]], axis=0)  # # 添加本次置信度最高的boundingbox的index

        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留交集小于一定阈值的boundingbox
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


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


def main():
    parser = ArgumentParser()
    # parser.add_argument('--config', default='./configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py',
    #                     help='config file')
    parser.add_argument('--config', default='./configs/mot/bytetrack/one_carafe_bytetrack_full_mdmt.py',
                        help='config file')

    parser.add_argument('--input', default='F:/A_Master_Menu/_A_dataset/MCMOT-new/MDMT/1/',
                        help='input video file or folder')

    parser.add_argument('--xml_dir', default='F:/A_Master_Menu/_A_dataset/MCMOT-new/new_xml/',
                        help='input xml file of the groundtruth')

    parser.add_argument('--result_dir', default='./json_resultfiles',
                        help='result_dir name, no "/" in the end')
    parser.add_argument('--method', default='Firstframe_initialized_faster_rcnn_r50_fpn_carafe_1x_full_mdmt',
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
        # print(args.checkpoint)
        model = init_model(args.config, args.checkpoint, device=args.device)
        model2 = init_model(args.config, args.checkpoint, device=args.device)

        prog_bar = mmcv.ProgressBar(len(imgs))

        matched_ids = []
        A_max_id = 0
        B_max_id = 0

        result_dict = {}
        result_dict2 = {}
        supplement_dict = {}
        supplement_dict2 = {}
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
            
            
            if len(track_bboxes) != 0:
                A_max_id = max(A_max_id, max(track_bboxes[:, 0]))
                B_max_id = max(B_max_id, max(track_bboxes2[:, 0]))
            
            # #############NMS##########################NMS##########################NMS##########
            # thresh = 0.3
            # # print(len(track_bboxes))
            # track_bboxes = all_nms(track_bboxes, thresh)
            # # print("2", track_bboxes)
            # track_bboxes2 = all_nms(track_bboxes2, thresh)
            # #############NMS##########################NMS##########################NMS##########
            

            # # 通过网络输出得到追踪框，并写入字典，准备后续的文件输出
            # track_bboxes = np.concatenate(result1['track_bboxes'][:], axis=0)
            # track_bboxes2 = np.concatenate(result2['track_bboxes'][:], axis=0)
            # # print(len(result1['track_bboxes']))
            
            # 更新result
            result['track_bboxes'][0] = track_bboxes
            result2['track_bboxes'][0] = track_bboxes2

            # '''
            bboxes1 = torch.tensor(track_bboxes[:, 1:5], dtype=torch.long)
            ids1 = torch.tensor(track_bboxes[:, 0], dtype=torch.long)
            labels1 = torch.zeros_like(torch.tensor(track_bboxes[:, 0]))
            # labels1 = torch.tensor(track_bboxes[:, 0])

            bboxes2 = torch.tensor(track_bboxes2[:, 1:5], dtype=torch.long)
            ids2 = torch.tensor(track_bboxes2[:, 0], dtype=torch.long)
            labels2 = torch.zeros_like(torch.tensor(track_bboxes2[:, 0]))
            # labels2 = torch.zeros_like(torch.tensor(track_bboxes2[:, 0]))

            result_dict["frame={}".format(i)] = track_bboxes[:, 0:5].tolist()
            result_dict2["frame={}".format(i)] = track_bboxes2[:, 0:5].tolist()

            if args.output is not None:
                if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path, f'{i:06d}.jpg')
                    out_file2 = osp.join(out_path2, f'{i:06d}.jpg')
                    # out_file3 = osp.join(out_path3, f'{i:06d}.jpg')
                    # out_file4 = osp.join(out_path4, f'{i:06d}.jpg')
                else:
                    out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
                    out_file2 = osp.join(out_path2, img2.rsplit(os.sep, 1)[-1])
                    # out_file3 = osp.join(out_path3, img3.rsplit(os.sep, 1)[-1])
                    # out_file4 = osp.join(out_path4, img4.rsplit(os.sep, 1)[-1])
            else:
                out_file = None
                out_file2 = None
                # out_file3 = None
                # out_file4 = None
            model.show_result(
                img,
                result,
                score_thr=args.score_thr,
                show=args.show,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file,
                backend=args.backend)
            model2.show_result(
                img2,
                result2,
                score_thr=args.score_thr,
                show=args.show,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file2,
                backend=args.backend)
            prog_bar.update()
            
        time_end = time.time()
        method = args.method
        # method = "NMS-one_carafe_bytetrack_full_mdmt"
        json_dir = "{}/{}/".format(args.result_dir, method)

        # method = "Firstframe_initialized_faster_rcnn_r50_fpn_carafe_1x_full_mdmt"
        # method = "Firstframe_initialized_bytetrack_autoassign_full_mdmt-private-half"

        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        with open("{0}/{1}.json".format(json_dir, sequence1), "w") as f:
            json.dump(result_dict, f, indent=4)
            print("输出文件A写入完成！")
        with open("{0}/{1}.json".format(json_dir, sequence2), "w") as f2:
            json.dump(result_dict2, f2, indent=4)
            print("输出文件B写入完成！")
        with open("{0}/time.txt".format(json_dir), "a") as f3:
            f3.write("{} time consume :{}\n".format(sequence1, time_end-time_start))
            print("输出文件time.txt写入完成！")

        if args.output and OUT_VIDEO:
            print(f'making the output video at {args.output} with a FPS of {fps}')
            mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
            mmcv.frames2video(out_path2, args.output2, fps=fps, fourcc='mp4v')
            # mmcv.frames2video(out_path3, args.output3, fps=fps, fourcc='mp4v')
            # mmcv.frames2video(out_path4, args.output4, fps=fps, fourcc='mp4v')
            out_dir.cleanup()
            out_dir2.cleanup()
            # out_dir3.cleanup()
            # out_dir4.cleanup()
    time_end_all = time.time()
    with open("{0}/time.txt".format(json_dir), "a") as f3:
        f3.write("ALL time consume :{}\n".format(time_end_all-time_start_all))


if __name__ == '__main__':
    main()
#     # 启动参数（指浏览器与百度搜索内容）
#     lists = ["test1", "test2", "test3", "test4", "test5", "test6"]
#     threads = []
#     files = range(len(lists))
    
#     for test_dir_ in lists:
#         print(test_dir_)
#         t = Thread(target=main, args=(test_dir_, test_dir_))
#         threads.append(t)
#     # 启动线程
#     for t in files:
#         threads[t].start()
        
#     for t in files:
#         threads[t].join()  # join()函数必须放在线程启动之后
