# _*_ coding: utf-8 _*_
# @Time    :2022/7/12 16:38
# @Author  :LiuZhihao
# @File    :supplement.py

# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv

from mmtrack.apis import inference_mot, init_model
import cv2
import xml.etree.ElementTree as ET
import torch
import numpy as np
import json
import time
from utils.matching_pure import matching, calculate_cent_corner_pst
from utils.common import read_xml_r, all_nms, get_matched_ids_frame1, get_matched_ids, A_same_target_refresh_same_ID, B_same_target_refresh_same_ID, same_target_refresh_same_ID
from utils.trans_matrix import global_compute_transf_matrix as compute_transf_matrix


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

    parser.add_argument('--result_dir', default='./json_resultfiles2/multiDrone_Globalmatching-NMS',
                        help='result_dir name, no "/" in the end')
    parser.add_argument('--method', default='NMS-one_carafe_bytetrack_full_mdmt',
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

        time_start = time.time()
        # test and show/save the images
        for i, img in enumerate(imgs):
            flag = 0
            coID_confirme = []

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

            # ############NMS##########################NMS##########################NMS##########
            # thresh = 0.3
            # # print(len(track_bboxes))
            # track_bboxes = all_nms(track_bboxes, thresh)
            # # print("2", track_bboxes)
            # track_bboxes2 = all_nms(track_bboxes2, thresh)
            # #############NMS##########################NMS##########################NMS##########

            result_dict["frame={}".format(i)] = track_bboxes[:, 0:5].tolist()
            result_dict2["frame={}".format(i)] = track_bboxes2[:, 0:5].tolist()
            # '''
            # 计算追踪目标中心点，进而计算两机间变换矩阵
            cent_allclass, corner_allclass = calculate_cent_corner_pst(image1,
                                                                       track_bboxes)  # corner_allclass:ndarray 2n*2
            cent_allclass2, corner_allclass2 = calculate_cent_corner_pst(image2,
                                                                         track_bboxes2)  # cent_allclass:ndarray n*2

            # 第一帧：
            if i == 0:
                # 遍历两个track_bboxes，计算匹配点和matchedID
                pts_src, pts_dst, matched_ids = get_matched_ids_frame1(track_bboxes, track_bboxes2, cent_allclass,
                                                                       cent_allclass2)
                A_max_id = max(A_max_id, max(track_bboxes[:, 0]))
                B_max_id = max(B_max_id, max(track_bboxes2[:, 0]))
                # 计算变换矩阵f
                if len(pts_src) >= 5:
                    f1, status1 = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
                    f2, status2 = cv2.findHomography(pts_dst, pts_src, cv2.RANSAC, 5.0)
                    # print(f1, f2)
                    f1_last = f1
                    f2_last = f2
                    # 图像融合可视化
                    '''
                    leftImage = image1
                    rightImage = image2
                    resulttt = cv2.warpPerspective(leftImage, f1, (leftImage.shape[1] + rightImage.shape[1], leftImage.shape[0] + rightImage.shape[0]))
                    # 融合方法1
                    resulttt.astype(np.float32)
                    resulttt = resulttt / 2
                    resulttt[0:rightImage.shape[0], 0:rightImage.shape[1]] += rightImage / 2
                    cv2.imwrite("./matchingimages/matching{}.jpg".format(i), resulttt)
                    '''
                # sift 全局匹配（试验）
                # matching(image1, image2, cent_allclass, corner_allclass, 1)
                # '''
                if args.output is not None:
                    if IN_VIDEO or OUT_VIDEO:
                        out_file = osp.join(out_path, f'{i:06d}.jpg')
                        out_file2 = osp.join(out_path2, f'{i:06d}.jpg')
                    else:
                        out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
                        out_file2 = osp.join(out_path2, img.rsplit(os.sep, 1)[-1])
                else:
                    out_file = None
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
                # '''
                prog_bar.update()
                continue
            ###########################################################################################################################
            # 第一帧结束，后续帧通过MOT模型进行跟踪，对新产生的ID通过旋转矩阵进行双机匹配（旋转矩阵通过上一帧已匹配目标计算）
            # matched_ids
            # 遍历两个track_bboxes，计算匹配点和matchedID,并得到新增ID及其中心点

            matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, \
            A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, B_old_not_matched_pts_corner \
                = get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass,
                                  corner_allclass2, A_max_id, B_max_id, coID_confirme)
            # print(track_bboxes[:, 0])
            # print(sorted(matched_ids))
            # if len(track_bboxes) != 0:
            #     A_max_id = max(A_max_id, max(track_bboxes[:, 0]))
            #     B_max_id = max(B_max_id, max(track_bboxes2[:, 0]))

            # 计算变换矩阵f
            # print(len(pts_src))
            f1, f1_last = compute_transf_matrix(pts_src, pts_dst, f1_last, image1, image2)
            # '''
            # 图像融合可视化
            leftImage = image1
            rightImage = image2
            resulttt = cv2.warpPerspective(leftImage, f1, (
            leftImage.shape[1] + rightImage.shape[1], leftImage.shape[0] + rightImage.shape[0]))
            resulttt.astype(np.float32)
            resulttt = resulttt / 2
            resulttt[0:rightImage.shape[0], 0:rightImage.shape[1]] += rightImage / 2
            cv2.imwrite("./matchingimages/matching{}-1.jpg".format(i), resulttt)

            # '''
            # 进行ID更改，将能匹配的新目标赋予同ID(选取旧ID作为关联ID)
            track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme = \
                A_same_target_refresh_same_ID(A_new_ID, A_pts, A_pts_corner, f1, cent_allclass2, track_bboxes,
                                              track_bboxes2, matched_ids, det_bboxes, det_bboxes2, image2,
                                              coID_confirme, thres=80)
            if flag == 1:
                print("flag == 1")
                flag = 0
            # 更新result
            result['track_bboxes'][0] = track_bboxes
            result2['track_bboxes'][0] = track_bboxes2
            ####################################################222###################################################
            matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, \
            A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, B_old_not_matched_pts_corner \
                = get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass,
                                  corner_allclass2, A_max_id, B_max_id, coID_confirme)
            f2, f2_last = compute_transf_matrix(pts_dst, pts_src, f2_last, image2, image1)
            # 图像融合可视化
            leftImage = image2
            rightImage = image1
            resulttt = cv2.warpPerspective(leftImage, f2, (
                leftImage.shape[1] + rightImage.shape[1], leftImage.shape[0] + rightImage.shape[0]))
            resulttt.astype(np.float32)
            resulttt = resulttt / 2
            resulttt[0:rightImage.shape[0], 0:rightImage.shape[1]] += rightImage / 2
            cv2.imwrite("./matchingimages/matching{}-2.jpg".format(i), resulttt)

            # 进行ID更改，将能匹配的新目标赋予同ID(选取旧ID作为关联ID)
            track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme = \
                B_same_target_refresh_same_ID(B_new_ID, B_pts, B_pts_corner, f2, cent_allclass, track_bboxes,
                                              track_bboxes2, matched_ids, det_bboxes, det_bboxes2, image1,
                                              coID_confirme, thres=80)
            ##################3##################3##################3##################3
            # 不是新目标，且没有匹配上的旧目标，在此处再计算一下看有没有能对的上的ID，防止在新目标出现时没对上那么后续就再也对不上的情况（对比试验/消融实验？？？？）
            matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, \
                A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, B_old_not_matched_pts_corner \
                = get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass,
                                  corner_allclass2, A_max_id, B_max_id, coID_confirme)
            # # 不是新目标，且没有匹配上的旧目标，在此处再计算一下看有没有能对的上的ID，防止在新目标出现时没对上那么后续就再也对不上的情况（对比试验/消融实验？？？？）
            track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme = same_target_refresh_same_ID(A_old_not_matched_ids,
                                                                                         A_old_not_matched_pts,
                                                                                         A_old_not_matched_pts_corner,
                                                                                         f1,
                                                                                         cent_allclass2, track_bboxes,
                                                                                         track_bboxes2, matched_ids,
                                                                                         det_bboxes, det_bboxes2,
                                                                                         image2, coID_confirme, thres=50)
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if len(track_bboxes) != 0:
                A_max_id = max(A_max_id, max(track_bboxes[:, 0]))
                B_max_id = max(B_max_id, max(track_bboxes2[:, 0]))
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ############NMS##########################NMS##########################NMS##########
            thresh = 0.3
            # print(len(track_bboxes))
            track_bboxes = all_nms(track_bboxes, thresh)
            # print("2", track_bboxes)
            track_bboxes2 = all_nms(track_bboxes2, thresh)
            #############NMS##########################NMS##########################NMS##########




            # if flag == 1:
            #     print("flag == 1")
            #     cv2.rectangle(image2, (int(track_bboxes2[-1, 1]), int(track_bboxes2[-1, 2])), (int(track_bboxes2[-1, 3]), int(track_bboxes2[-1, 4])), (250, 33, 32), 5)
            #     cv2.imshow("fksoadf", image2)
            #     cv2.waitKey(1000)
            #     flag = 0

            # 更新result
            result['track_bboxes'][0] = track_bboxes
            result2['track_bboxes'][0] = track_bboxes2
            track_bboxes_old = track_bboxes.copy()
            track_bboxes2_old = track_bboxes2.copy()

            # '''
            # print(track_bboxes)
            bboxes1 = torch.tensor(track_bboxes[:, 1:5], dtype=torch.long)
            ids1 = torch.tensor(track_bboxes[:, 0], dtype=torch.long)
            labels1 = torch.zeros_like(torch.tensor(track_bboxes[:, 0]))
            # labels1 = torch.tensor(track_bboxes[:, 0])

            bboxes2 = torch.tensor(track_bboxes2[:, 1:5], dtype=torch.long)
            ids2 = torch.tensor(track_bboxes2[:, 0], dtype=torch.long)
            labels2 = torch.zeros_like(torch.tensor(track_bboxes2[:, 0]))
            # labels2 = torch.zeros_like(torch.tensor(track_bboxes2[:, 0]))
            #####################################################################

            result_dict["frame={}".format(i)] = track_bboxes[:, 0:5].tolist()
            result_dict2["frame={}".format(i)] = track_bboxes2[:, 0:5].tolist()
            # print("result_dict", result_dict)

            # '''
            if args.output is not None:
                if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path, f'{i:06d}.jpg')
                    out_file2 = osp.join(out_path2, f'{i:06d}.jpg')
                else:
                    out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
                    out_file2 = osp.join(out_path2, img.rsplit(os.sep, 1)[-1])
            else:
                out_file = None
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
            # '''
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
            f3.write("{} time consume :{}\n".format(sequence1, time_end-time_start))
            print("输出文件time.txt写入完成！")

        # '''
        if args.output and OUT_VIDEO:
            print(f'making the output video at {args.output} with a FPS of {fps}')
            mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
            mmcv.frames2video(out_path2, args.output2, fps=fps, fourcc='mp4v')
            out_dir.cleanup()
        # '''
    time_end_all = time.time()
    with open("{0}/time.txt".format(json_dir), "a") as f3:
        f3.write("ALL time consume :{}\n".format(time_end_all-time_start_all))



if __name__ == '__main__':
    main()

# import json
# with open("./resultfiles/-34-1.json") as f:
#     load_json = json.load(f)
# print(load_json["frame=0"])