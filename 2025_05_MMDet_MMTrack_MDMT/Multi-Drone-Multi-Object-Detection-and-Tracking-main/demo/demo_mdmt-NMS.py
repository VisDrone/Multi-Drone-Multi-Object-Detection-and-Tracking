# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import copy
from mmtrack.apis import inference_mot, init_model # ,inference_reid_mdmt_com  # ,inference_reid_mdmt
import pynvml
import torch
import os 
import json
import numpy as np
import time

from threading import Thread


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

def main():
# def main(test_dir, test_dir2):
    parser = ArgumentParser()
    # parser.add_argument('--config', help='config file',default = 'configs/mot/bytetrack/bytetrack_yolox_x_mdmt-private-half.py')
    # parser.add_argument('--checkpoint', help='checkpoint file',default = 'checkpoints/best_bbox_mAP_epoch_67.pth')
    # parser.add_argument('--checkpoint2', help='checkpoint file',default = 'checkpoints/best_bbox_mAP_epoch_67.pth')

    #############################################
    # parser.add_argument('--config', default='configs/mot/bytetrack/one_carafe_bytetrack_full_mdmt.py', help='config file')
    # parser.add_argument('--checkpoint', help='checkpoint file',default = 'checkpoint/faster_rcnn_r50_fpn_carafe_1x_full_mdmt/epoch_12.pth')
    # parser.add_argument('--checkpoint2', help='checkpoint file',default = 'checkpoint/faster_rcnn_r50_fpn_carafe_1x_full_mdmt/epoch_12.pth')
    #############################################

    parser.add_argument('--config', help='config file',default = 'configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py')
    parser.add_argument('--checkpoint', help='checkpoint file',default = 'checkpoint/autoassign_r50_fpn_8x2_1x_full_mdmt/epoch_60.pth')
    parser.add_argument('--checkpoint2', help='checkpoint file',default = 'checkpoint/autoassign_r50_fpn_8x2_1x_full_mdmt/epoch_60.pth')

    parser.add_argument('--reid_config', help='config file',default = 'configs/mot/auavreid/resnet50_b32x8_FULL_MDMT.py')
    parser.add_argument('--input', help='input video file or folder', default = '../autodl-tmp/test/') # .format(test_dir))
    # parser.add_argument('--input2', help='input video file or folder',default = 'data/FULL_MDMT/test/26-2/img1/')
    # parser.add_argument('--input', help='input video file or folder',default = 'result/video/26-1.mp4')
    # parser.add_argument('--input2', help='input video file or folder',default = 'result/video/26-2.mp4')

    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder',default = 'result/full_mdmt_bytetrack/26-1/')
    parser.add_argument(
        '--output2', help='output video file (mp4 format) or folder',default = 'result/full_mdmt_bytetrack/26-2/')
    parser.add_argument(
        '--output3', help='output video file (mp4 format) or folder',default = 'result/full_mdmt_bytetrack/26-com/')
    parser.add_argument(
        '--output4', help='output video file (mp4 format) or folder',default = 'result/full_mdmt_bytetrack/26-com2/')

    # parser.add_argument('--config', help='config file',default = 'configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py')
    # parser.add_argument('--reid_config', help='config file',default = 'configs/mot/auavreid/resnet50_b32x8_MDMT.py')
    # parser.add_argument('--input', help='input video file or folder',default = 'data/MOT17/test/MOT17-01-FRCNN/img1/')
    # parser.add_argument('--input2', help='input video file or folder',default = 'data/MOT17/test/MOT17-01-DPM/img1/')
    # parser.add_argument(
    #     '--output', help='output video file (mp4 format) or folder',default = 'result/MOT17-01-FRCNN.mp4')
    # parser.add_argument(
    #     '--output2', help='output video file (mp4 format) or folder',default = 'result/MOT17-01-DPM.mp4')

    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video',default = 12)
    args = parser.parse_args()
    assert args.output or args.show
    # load images
    
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
        print(os.path.join(args.input+dirrr+"/" +"img1"+ "/"))
        sequence_dir = os.path.join(args.input + dirrr + "/" + "img1"+ "/")

        if osp.isdir(sequence_dir):
            imgs = sorted(
                filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                    os.listdir(sequence_dir)),
                key=lambda x: int(x.split('.')[0]))
            IN_VIDEO = False
        else:
            imgs = mmcv.VideoReader(sequence_dir)
            IN_VIDEO = True
        # if osp.isdir(args.input2):
        #     imgs2 = sorted(
        #         filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
        #             os.listdir(args.input2)),
        #         key=lambda x: int(x.split('.')[0]))
        #     IN_VIDEO = False
        # else:
        #     imgs2 = mmcv.VideoReader(args.input2)
        #     IN_VIDEO = True
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
        # if args.output3 is not None:
        #     if args.output3.endswith('.mp4'):
        #         OUT_VIDEO = True
        #         out_dir3 = tempfile.TemporaryDirectory()
        #         out_path3 = out_dir3.name
        #         _out3 = args.output3.rsplit(os.sep, 1)
        #         if len(_out3) > 1:
        #             os.makedirs(_out3[0], exist_ok=True)
        #     else:
        #         OUT_VIDEO = False
        #         out_path3 = args.output3
        #         os.makedirs(out_path3, exist_ok=True)
        # if args.output4 is not None:
        #     if args.output4.endswith('.mp4'):
        #         OUT_VIDEO = True
        #         out_dir4 = tempfile.TemporaryDirectory()
        #         out_path4 = out_dir4.name
        #         _out4 = args.output4.rsplit(os.sep, 1)
        #         if len(_out4) > 1:
        #             os.makedirs(_out4[0], exist_ok=True)
        #     else:
        #         OUT_VIDEO = False
        #         out_path4 = args.output4
        #         os.makedirs(out_path4, exist_ok=True)
        fps = args.fps
        if args.show or OUT_VIDEO:
            if fps is None and IN_VIDEO:
                fps = imgs.fps
            if not fps:
                raise ValueError('Please set the FPS for the output video.')
            fps = int(fps)


        # build the model from a config file and a checkpoint file
        model = init_model(args.config, args.checkpoint, device=args.device)
        model2 = init_model(args.config, args.checkpoint2, device=args.device)
        # reid_model = init_model(args.reid_config, args.checkpoint, device=args.device)
        # reid_model = init_model(args.reid_config,  device=args.device)

        prog_bar = mmcv.ProgressBar(len(imgs))
        # test and show/save the images
        num_picture=len(imgs)
        # num_imgs2=len(imgs2)
        # num_picture= len(num_imgs1)

        result_dict = {}
        result_dict2 = {}
        time_start = time.time()
        # for i, img in enumerate(imgs):
        for i in range(0, num_picture):
            img = imgs[i]
            # img2 = imgs2[i]
            # img3 = copy.copy(imgs[i])
            # img4 = copy.copy(imgs2[i])
            # img2 = img1
            if isinstance(img, str):
                img = osp.join(sequence_dir, img)
            # if isinstance(img2, str):
                img2 = img.replace("-1", "-2")
            # if isinstance(img3, str):
            #     img3 = osp.join(args.input, img3)
            # if isinstance(img4, str):
            #     img4 = osp.join(args.input2, img4)
            sequence1 = img.split("/")[-3]
            sequence2 = img2.split("/")[-3]
            result1 = inference_mot(model, img, frame_id=i)
            result2 = inference_mot(model2, img2, frame_id=i)

            # 通过网络输出得到追踪框，并写入字典，准备后续的文件输出
            track_bboxes = np.concatenate(result1['track_bboxes'][:], axis=0)
            track_bboxes2 = np.concatenate(result2['track_bboxes'][:], axis=0)
            # print(len(result1['track_bboxes']))

            # #############NMS##########################NMS##########################NMS##########
            thresh = 0.3
            # print(len(track_bboxes))
            track_bboxes = all_nms(track_bboxes, thresh)
            # print("2", track_bboxes)
            track_bboxes2 = all_nms(track_bboxes2, thresh)
            # #############NMS##########################NMS##########################NMS##########

            result_dict["frame={}".format(i)] = track_bboxes[:, 0:5].tolist()
            result_dict2["frame={}".format(i)] = track_bboxes2[:, 0:5].tolist()


            # result3 = inference_reid_mdmt_com(reid_model, img1=img, frame_id1=i, result1=result1,img2=img2,frame_id2=i,result2=result2)
            # result4 = inference_reid_mdmt_com(reid_model, img1=img2, frame_id1=i, result1=result2,img2=img,frame_id2=i,result2=result1)
            # print("第 ",i," 次结束 ")
            # print()
            # print(reid_model)
            # print('result1 = ', result1['det_bboxes'])
            # print('result2 = ', result2['det_bboxes'])
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
                result1,
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
            # reid_model.show_result(
            #     img3,
            #     result3,
            #     score_thr=args.score_thr,
            #     show=args.show,
            #     wait_time=int(1000. / fps) if fps else 0,
            #     out_file=out_file3,
            #     backend=args.backend)
            # reid_model.show_result(
            #     img4,
            #     result4,
            #     score_thr=args.score_thr,
            #     show=args.show,
            #     wait_time=int(1000. / fps) if fps else 0,
            #     out_file=out_file4,
            #     backend=args.backend)
            prog_bar.update()
            
        time_end = time.time()
        method = "NMS-autoassign_r50_fpn_8x2_1x_full_mdmt"
        json_dir = "./json_resultfiles/{}/".format(method)
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
