# _*_ coding: utf-8 _*_
# @Time    :2022/12/7 16:38
# @Author  :LiuZhihao
# @File    :common.py

import numpy as np
import cv2
import xml.etree.ElementTree as ET
import torch
from matching_pure import matching, calculate_cent_corner_pst


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


def calculate_cent_corner_pst(img1, result1):
    cent_allclass = []
    corner_allclass = []

    # for calss_num, result in enumerate(result1):
    center_pst = np.array([])
    corner_pst = np.array([])
    for dots in result1:
        # print("dots:", dots)
        x1 = dots[1]
        y1 = dots[2]
        x2 = dots[3]
        y2 = dots[4]
        centx = (x1 + x2) / 2
        centy = (y1 + y2) / 2
        # 收集检测结果的中点和角点
        if center_pst.size == 0:
            center_pst = np.array([[centx, centy]])
        else:
            center_pst = np.append(center_pst, [[centx, centy]], axis=0)  # 前缀pst=和axis = 0一定要有!!!!!!!!!!
        if corner_pst.size == 0:
            corner_pst = np.array([[x1, y1],
                                   [x2, y2]])
        else:
            corner_pst = np.append(corner_pst, [[x1, y1], [x2, y2]], axis=0)  # 前缀pst=和axis = 0一定要有!!!!!!!!!!
        # cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 33, 32), 5)
    # center_pst = center_pst.reshape(-1, 2).astype(np.float32)
    # corner_pst = corner_pst.reshape(-1,  2).astype(np.float32)

    # cent_allclass.append(center_pst)
    # corner_allclass.append(corner_pst)
    # cv2.imshow("img1", img1)
    # cv2.waitKey(100)

    return center_pst, corner_pst


# 遍历两个trackbox,统计同ID:
def get_matched_ids_frame1(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2):
    matched_ids_cache = []
    pts_src = []
    pts_dst = []
    for m, dots in enumerate(track_bboxes):
        trac_id = dots[0]
        # A_max_id = trac_id if A_max_id<trac_id else A_max_id
        for n, dots2 in enumerate(track_bboxes2):
            trac2_id = dots2[0]
            # B_max_id = trac2_id if B_max_id<trac2_id else B_max_id
            if trac_id == trac2_id:
                # cv2.circle(image1, (int(cent_allclass[m][0]), int(cent_allclass[m][1])), 30, (0, 345, 255))
                # cv2.imshow('img', image1)
                # cv2.waitKey(100)
                # cv2.circle(image2, (int(cent_allclass2[n][0]), int(cent_allclass2[n][1])), 30, (54, 0, 255))
                # cv2.imshow('img2', image2)
                # cv2.waitKey(10)
                # 将匹双机配点中心放入列表里面，后续用来计算旋转矩阵
                pts_src.append(cent_allclass[m])
                pts_dst.append(cent_allclass2[n])
                matched_ids_cache.append(trac_id)
                break

    # 变换数据格式，用于后续计算变换矩阵
    pts_src = np.array(pts_src)
    pts_dst = np.array(pts_dst)
    matched_ids = matched_ids_cache.copy()
    return pts_src, pts_dst, matched_ids


def get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass, corner_allclass2,
                    A_max_id, B_max_id, coID_confirme):
    matched_ids_cache = []
    A_new_ID = []
    B_new_ID = []
    A_pts = []
    B_pts = []
    A_pts_corner = []
    B_pts_corner = []
    A_old_not_matched_ids = []
    B_old_not_matched_ids = []
    A_old_not_matched_pts = []
    B_old_not_matched_pts = []
    A_old_not_matched_pts_corner = []
    B_old_not_matched_pts_corner = []

    pts_src = []
    pts_dst = []
    for m, dots in enumerate(track_bboxes):
        trac_id = dots[0]
        if trac_id in coID_confirme:
            print("trac_id", trac_id)
            matched_ids_cache.append(trac_id)
            pts_src.append(cent_allclass[m])
            print("cent_allclass[m]", cent_allclass[m])
            for n, dots2 in enumerate(track_bboxes2):
                trac2_id = dots2[0]
                if trac2_id == trac_id:
                    pts_dst.append(cent_allclass2[n])
                    print("cent_allclass2[n]", cent_allclass2[n])
            continue
        if A_max_id < trac_id:
            if trac_id not in A_new_ID:
                A_new_ID.append(trac_id)
                A_pts.append(cent_allclass[m])
                A_pts_corner.append([corner_allclass[2 * m], corner_allclass[2 * m + 1]])
            continue
        for n, dots2 in enumerate(track_bboxes2):
            trac2_id = dots2[0]
            flag_matched = 0
            if B_max_id < trac2_id:
                if trac2_id not in B_new_ID:
                    B_new_ID.append(trac2_id)
                    B_pts.append(cent_allclass2[n])
                    B_pts_corner.append([corner_allclass2[2 * n], corner_allclass2[2 * n + 1]])
                continue
            if trac_id == trac2_id:
                # cv2.circle(image1, (int(cent_allclass[m][0]), int(cent_allclass[m][1])), 30, (0, 345, 255))
                # cv2.imshow('img', image1)
                # cv2.waitKey(100)
                # cv2.circle(image2, (int(cent_allclass2[n][0]), int(cent_allclass2[n][1])), 30, (54, 0, 255))
                # cv2.imshow('img2', image2)
                # cv2.waitKey(10)
                # 将匹双机配点中心放入列表里面，后续用来计算旋转矩阵
                matched_ids_cache.append(trac_id)
                pts_src.append(cent_allclass[m])
                pts_dst.append(cent_allclass2[n])
                flag_matched = 1
                break
        if flag_matched == 0 and trac_id not in A_old_not_matched_ids:
                A_old_not_matched_ids.append(trac_id)
                A_old_not_matched_pts.append(cent_allclass[m])
                A_old_not_matched_pts_corner.append([corner_allclass[2 * m], corner_allclass[2 * m + 1]])

    for n, dots2 in enumerate(track_bboxes2):
        trac2_id = dots2[0]
    if trac2_id not in B_old_not_matched_ids and trac2_id not in matched_ids_cache and trac2_id not in B_new_ID:
        B_old_not_matched_ids.append(trac2_id)
        B_old_not_matched_pts.append(cent_allclass2[n])
        B_old_not_matched_pts_corner.append([corner_allclass2[2 * n], corner_allclass2[2 * n + 1]])

            # else:  # A中旧目标没配对的也加入匹配序列
            #     if trac_id not in A_new_ID:
            #         A_new_ID.append(trac_id)
            #         A_pts.append(cent_allclass[m])
            #         A_pts_corner.append([corner_allclass[2 * m], corner_allclass[2 * m + 1]])
    matched_ids = matched_ids_cache.copy()
    # 变换数据格式，用于后续计算变换矩阵
    pts_src = np.array(pts_src)
    pts_dst = np.array(pts_dst)
    return matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, B_old_not_matched_pts_corner



def A_same_target_refresh_same_ID(A_new_ID_func, A_pts_func, A_pts_corner_func, f1_func, cent_allclass2_func,
                                  track_bboxes_func, track_bboxes2_func, matched_ids_func, det_bboxes, det_bboxes2,
                                  image2, coID_confirme, thres=100):
    flag = 0
    IOU_flag = 0
    if len(A_pts_func) == 0:
        pass
    else:
        A_pts_func = np.array(A_pts_func).reshape(-1, 1, 2).astype(np.float32)
        A_pts_corner_func = np.array(A_pts_corner_func).reshape(-1, 1, 2).astype(np.float32)
        A_dst_func = cv2.perspectiveTransform(A_pts_func, f1_func)
        A_dst_corner_func = cv2.perspectiveTransform(A_pts_corner_func, f1_func)
        # print(A_pts, A_dst)
        # zipped = zip(A_dst, A_dst_corner)
        # for ii_class, (dst_cent, dst_corner) in enumerate(list(zipped)):
        if A_dst_func is not None:
            dist = np.zeros((len(A_dst_func), len(track_bboxes2_func)))
            # 以下为了计算valid，并补充到检测结果中##################3
            valid = []
            for ii, xy in enumerate(A_dst_func):
                min_x = min(A_dst_corner_func[ii * 2, 0, 0], A_dst_corner_func[ii * 2 + 1, 0, 0])
                max_x = max(A_dst_corner_func[ii * 2, 0, 0], A_dst_corner_func[ii * 2 + 1, 0, 0])
                min_y = min(A_dst_corner_func[ii * 2, 0, 1], A_dst_corner_func[ii * 2 + 1, 0, 1])
                max_y = max(A_dst_corner_func[ii * 2, 0, 1], A_dst_corner_func[ii * 2 + 1, 0, 1])
                if min_x > 1920 or max_x < 0 or min_y > 1080 or max_y < 0 or min_x < 0 or min_y < 0 or max_x > 1920 or max_y > 1080:
                    continue
                else:
                    for j, dots in enumerate(cent_allclass2_func):  # track_bboxes2  ndarray n*6
                        centx = int(dots[0])
                        centy = int(dots[1])
                        # cv2.circle(img2, (centx, centy), 9, (150, 34, 23), 3)
                        # cv2.circle(img2, (int(xy[0, 0]), int(xy[0, 1])), 9, (50, 340, 23), 3)

                        dist[ii, j] = ((xy[0, 0] - centx) ** 2 + (xy[0, 1] - centy) ** 2) ** 0.5
                        # cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)), (33, 33, 32), 5)
                    # 超参数！35
                    # print(ii, dist)
                    if min(dist[ii], default=0) < thres:
                        # A:A_index   B:B_index
                        # A_index = list(track_bboxes[:, 0]).index(A_new_ID[ii])
                        # B_index = list(dist[ii]).index(min(dist[ii], default=0))
                        # print(np.where(track_bboxes_func[:, 0] == A_new_ID_func[ii])[0])
                        A_index = np.where(track_bboxes_func[:, 0] == A_new_ID_func[ii])[0]# [0].astype(int)
                        B_index = np.where(dist[ii] == min(dist[ii]))[0]
                        # print(A_index)
                        # print(B_index)
                        if len(A_index) > 1:
                            A_index = min(A_index)  # #####################??????????????????????????????????????????min???
                        else:
                            A_index = A_index[0]
                        if len(B_index) > 1:
                            B_index = min(B_index)  # #####################??????????????????????????????????????????min???
                        else:
                            B_index = B_index[0]
                        print(A_index, B_index)
                        if track_bboxes2_func[B_index, 0] not in matched_ids_func:
                            # max_id = max(track_bboxes2[B_index, 0], A_new_ID[ii])
                            min_id = int(min(track_bboxes2_func[B_index, 0], A_new_ID_func[ii]))
                            track_bboxes_func[A_index, 0] = min_id
                            track_bboxes2_func[B_index, 0] = min_id
                            # 最重要的一行语句：
                            print("step1: coID confirme: {0} {1} to {2}".format(track_bboxes_func[A_index, 0],
                                                                         track_bboxes2_func[B_index, 0], min_id))
                            coID_confirme.append(int(min_id))
                            matched_ids_func.append(int(min_id))

    return track_bboxes_func, track_bboxes2_func, matched_ids_func, flag, coID_confirme


def B_same_target_refresh_same_ID(B_new_ID, B_pts, B_pts_corner, f2, cent_allclass,
                                  track_bboxes, track_bboxes2, matched_ids, det_bboxes, det_bboxes2,
                                  image2, coID_confirme, thres=150):
    flag = 0
    if len(B_pts) == 0:
        pass
    else:
        B_pts = np.array(B_pts).reshape(-1, 1, 2).astype(np.float32)
        B_pts_corner = np.array(B_pts_corner).reshape(-1, 1, 2).astype(np.float32)
        B_dst = cv2.perspectiveTransform(B_pts, f2)
        B_dst_corner = cv2.perspectiveTransform(B_pts_corner, f2)
        # print(A_pts, A_dst)
        # zipped = zip(A_dst, A_dst_corner)
        # for ii_class, (dst_cent, dst_corner) in enumerate(list(zipped)):
        if B_pts is not None:
            dist = np.zeros((len(B_pts), len(track_bboxes)))
            # 以下为了计算valid，并补充到检测结果中##################3
            valid = []
            for ii, xy in enumerate(B_dst):
                min_x = min(B_dst_corner[ii * 2, 0, 0], B_dst_corner[ii * 2 + 1, 0, 0])
                max_x = max(B_dst_corner[ii * 2, 0, 0], B_dst_corner[ii * 2 + 1, 0, 0])
                min_y = min(B_dst_corner[ii * 2, 0, 1], B_dst_corner[ii * 2 + 1, 0, 1])
                max_y = max(B_dst_corner[ii * 2, 0, 1], B_dst_corner[ii * 2 + 1, 0, 1])
                if min_x > 1920 or max_x < 0 or min_y > 1080 or max_y < 0 or min_x < 0 or min_y < 0 or max_x > 1920 or max_y > 1080:
                    continue
                else:
                    for j, dots in enumerate(cent_allclass):  # track_bboxes2  ndarray n*6
                        centx = int(dots[0])
                        centy = int(dots[1])
                        # cv2.circle(img2, (centx, centy), 2, (ii_class*50,34,23), 3)
                        # cv2.circle(img2, (int(xy[0, 0]), int(xy[0, 1])), 2, (ii_class*50,34,23), 3)

                        dist[ii, j] = ((xy[0, 0] - centx) ** 2 + (xy[0, 1] - centy) ** 2) ** 0.5
                        # cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)), (33, 33, 32), 5)
                    # 超参数！35
                    # print(ii, dist)
                    if min(dist[ii], default=0) < thres:

                        # A:A_index   B:B_index
                        # A_index = list(track_bboxes[:, 0]).index(A_new_ID[ii])
                        # B_index = list(dist[ii]).index(min(dist[ii], default=0))
                        B_index = np.where(track_bboxes2[:, 0] == B_new_ID[ii])[0]
                        A_index = np.where(dist[ii] == min(dist[ii]))[0]
                        # print(A_index)
                        # print(B_index)
                        if len(A_index) > 1:
                            A_index = min(A_index)  # #####################??????????????????????????????????????????min???
                        else:
                            A_index = A_index[0]
                        if len(B_index) > 1:
                            B_index = min(B_index)  # #####################??????????????????????????????????????????min???
                        else:
                            B_index = B_index[0]
                        print(A_index, B_index)
                        # 需不需要加这个判断？？？？？？？？？？
                        if track_bboxes[A_index, 0] not in matched_ids:
                            # max_id = max(track_bboxes2[B_index, 0], A_new_ID[ii])
                            # print(track_bboxes[A_index, 0], B_new_ID[ii])
                            min_id = min(track_bboxes[A_index, 0], B_new_ID[ii])
                            track_bboxes[A_index, 0] = min_id
                            track_bboxes2[B_index, 0] = min_id
                            # 最重要的一行语句：
                            # print(min_id)
                            print("step2: coID confirme: {0} {1} to {2}".format(track_bboxes[A_index, 0],
                                                                         track_bboxes2[B_index, 0], min_id))
                            coID_confirme.append(int(min_id))
                            matched_ids.append(int(min_id))
                            # print("coID confirmeB_2_A:", min_id)
                        # else:
                    # else:
                    #     # 让映射框和检测框做匹配，匹配上的检测框做为补充框并且加入matched ids
                    #     for boxx in det_bboxes:
                    #         xmin1, ymin1, xmax1, ymax1 = min_x, min_y, max_x, max_y
                    #         xmin2, ymin2, xmax2, ymax2 = boxx[0], boxx[1], boxx[2], boxx[3]
                    #         xx1 = np.max([xmin1, xmin2])
                    #         yy1 = np.max([ymin1, ymin2])
                    #         xx2 = np.min([xmax1, xmax2])
                    #         yy2 = np.min([ymax1, ymax2])
                    #
                    #         area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                    #         area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                    #         inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                    #         iou = inter_area / (area1 + area2 - inter_area + 1e-6)
                    #         if iou >= 0.6:
                    #             print("supply A")
                    #             track_bboxes = np.concatenate((track_bboxes, np.array([[B_new_ID[ii], xmin2, ymin2, xmax2, ymax2, 0.999]])), axis=0)
                    #             matched_ids.append(int(B_new_ID[ii]))
                    # np.append(track_bboxes, [B_new_ID[ii], min_x, min_y, max_x, max_y, 0.999])

                    # valid.append([min(dst_corner[ii * 2, 0, 0], dst_corner[ii * 2 + 1, 0, 0]),
                    #               min(dst_corner[ii * 2, 0, 1], dst_corner[ii * 2 + 1, 0, 1]),
                    #               max(dst_corner[ii * 2, 0, 0], dst_corner[ii * 2 + 1, 0, 0]),
                    #               max(dst_corner[ii * 2, 0, 1], dst_corner[ii * 2 + 1, 0, 1]),
                    #               0.99])  # 左上右下点的确定
                    # cv2.rectangle(img2, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 33, 32), 5)

    return track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme


def same_target_refresh_same_ID(A_new_ID_func, A_pts_func, A_pts_corner_func, f1_func, cent_allclass2_func,
                                track_bboxes_func, track_bboxes2_func, matched_ids_func, det_bboxes, det_bboxes2,
                                image2, coID_confirme, thres=100):
    flag = 0
    IOU_flag = 0
    if len(A_pts_func) == 0:
        pass
    else:
        A_pts_func = np.array(A_pts_func).reshape(-1, 1, 2).astype(np.float32)
        A_pts_corner_func = np.array(A_pts_corner_func).reshape(-1, 1, 2).astype(np.float32)
        A_dst_func = cv2.perspectiveTransform(A_pts_func, f1_func)
        A_dst_corner_func = cv2.perspectiveTransform(A_pts_corner_func, f1_func)
        # print(A_pts, A_dst)
        # zipped = zip(A_dst, A_dst_corner)
        # for ii_class, (dst_cent, dst_corner) in enumerate(list(zipped)):
        if A_dst_func is not None:
            dist = np.zeros((len(A_dst_func), len(track_bboxes2_func)))
            # 以下为了计算valid，并补充到检测结果中##################3
            valid = []
            for ii, xy in enumerate(A_dst_func):
                min_x = min(A_dst_corner_func[ii * 2, 0, 0], A_dst_corner_func[ii * 2 + 1, 0, 0])
                max_x = max(A_dst_corner_func[ii * 2, 0, 0], A_dst_corner_func[ii * 2 + 1, 0, 0])
                min_y = min(A_dst_corner_func[ii * 2, 0, 1], A_dst_corner_func[ii * 2 + 1, 0, 1])
                max_y = max(A_dst_corner_func[ii * 2, 0, 1], A_dst_corner_func[ii * 2 + 1, 0, 1])
                # or min_x < 0 or min_y < 0 or max_x > 1920 or max_y > 1080这个判断是后来加的，因为图像在边界的标注有问题！！！！！！！！！！！！！！！
                if min_x > 1920 or max_x < 0 or min_y > 1080 or max_y < 0 or min_x < 0 or min_y < 0 or max_x > 1920 or max_y > 1080:
                    continue
                else:
                    for j, dots in enumerate(cent_allclass2_func):  # track_bboxes2  ndarray n*6
                        centx = int(dots[0])
                        centy = int(dots[1])
                        # cv2.circle(img2, (centx, centy), 9, (150, 34, 23), 3)
                        # cv2.circle(img2, (int(xy[0, 0]), int(xy[0, 1])), 9, (50, 340, 23), 3)

                        dist[ii, j] = ((xy[0, 0] - centx) ** 2 + (xy[0, 1] - centy) ** 2) ** 0.5
                        # cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)), (33, 33, 32), 5)
                    # 超参数！35
                    # print(ii, dist)
                    if min(dist[ii], default=0) < thres:
                        # A:A_index   B:B_index
                        # A_index = list(track_bboxes[:, 0]).index(A_new_ID[ii])
                        # B_index = list(dist[ii]).index(min(dist[ii], default=0))
                        # print(np.where(track_bboxes_func[:, 0] == A_new_ID_func[ii])[0])
                        A_index = np.where(track_bboxes_func[:, 0] == A_new_ID_func[ii])[0]  # [0].astype(int)
                        B_index = np.where(dist[ii] == min(dist[ii]))[0]
                        if len(A_index) > 1:
                            A_index = min(
                                A_index)  # #####################??????????????????????????????????????????min???
                        else:
                            A_index = A_index[0]
                        if len(B_index) > 1:
                            B_index = min(
                                B_index)  # #####################??????????????????????????????????????????min???
                        else:
                            B_index = B_index[0]
                        # if track_bboxes2_func[B_index, 0] in matched_ids_func:  # 比较大小取小

                        if track_bboxes2_func[B_index, 0] not in matched_ids_func:
                            ##################matchede id里面还可能有错误匹配，进行进一步激励计算取最近的加入matched id,另一个重新加入unmatched ids，在后续进行rematch
                            ##################matchede id里面还可能有错误匹配，进行进一步激励计算取最近的加入matched id,另一个重新加入unmatched ids，在后续进行rematch
                            ##################matchede id里面还可能有错误匹配，进行进一步激励计算取最近的加入matched id,另一个重新加入unmatched ids，在后续进行rematch
                            # max_id = max(track_bboxes2[B_index, 0], A_new_ID[ii])
                            min_id = int(min(track_bboxes2_func[B_index, 0], A_new_ID_func[ii]))
                            track_bboxes_func[A_index, 0] = min_id
                            track_bboxes2_func[B_index, 0] = min_id
                            # 最重要的一行语句：
                            # print("coID confirme:", min_id)
                            print("step3: coID confirme: {0} {1} to {2}".format(track_bboxes_func[A_index, 0],
                                                                         track_bboxes2_func[B_index, 0], min_id))
                            coID_confirme.append(int(min_id))
                            matched_ids_func.append(int(min_id))
                    # else:
                    #     IOU_flag = 0
                    #     iou = np.array([])
                    #     # 让映射框和检测框做匹配，匹配上的检测框做为补充框并且加入matched ids
                    #     for boxx in det_bboxes2:
                    #         # 保证补充狂在图像内，防止溢出边界
                    #         min_x = min_x if min_x > 0 else 0
                    #         min_y = min_y if min_y > 0 else 0
                    #         max_x = max_x if max_x < 1920 else 1920
                    #         max_y = max_y if max_y < 1080 else 1080
                    #         # 下面是计算两个框IOU的程序：
                    #         xmin1, ymin1, xmax1, ymax1 = min_x, min_y, max_x, max_y
                    #         xmin2, ymin2, xmax2, ymax2 = boxx[0], boxx[1], boxx[2], boxx[3]
                    #
                    #         xx1 = np.max([xmin1, xmin2])
                    #         yy1 = np.max([ymin1, ymin2])
                    #         xx2 = np.min([xmax1, xmax2])
                    #         yy2 = np.min([ymax1, ymax2])
                    #
                    #         area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                    #         area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                    #         inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                    #         iou = np.append(iou, inter_area / (area1 + area2 - inter_area + 1e-6))
                    #         # 如果两个框IOU大于一定阈值则进行检测框补充，否则直接进行映射狂补充
                    #     if max(iou) > 0.3:
                    #         # indexindex = np.where(iou == max(iou))[0].astype(int)
                    #         # boxx = det_bboxes2[indexindex][0]
                    #         # 选置信度高的
                    #         indexindex = np.where(iou > 0.3)
                    #
                    #         if len(indexindex[0]) == 1:
                    #             indexindex = np.where(iou == max(iou))[0].astype(int)
                    #             boxx = det_bboxes2[indexindex][0]
                    #         else:
                    #             print(max(det_bboxes2[indexindex][:, -1]))
                    #             index = np.where(
                    #                 np.array(det_bboxes2[indexindex][:, -1]) == max(det_bboxes2[indexindex][:, -1]))
                    #             # print(indexindex)
                    #             # print(index[0])
                    #             # print(indexindex[0][index[0]])
                    #             boxx = det_bboxes2[indexindex[0][index[0]]][0]
                    #         # print(boxx)
                    #         xmin2, ymin2, xmax2, ymax2 = boxx[0], boxx[1], boxx[2], boxx[3]
                    #         if (xmax2 - xmin2) < 20:
                    #             continue
                    #         if (ymax2 - ymin2) < 20:
                    #             continue
                    #         print("suppliment")
                    #         IOU_flag = 1
                    #         track_bboxes2_func = np.concatenate((track_bboxes2_func, np.array(
                    #             [[A_new_ID_func[ii], xmin2, ymin2, xmax2, ymax2, 0.999]])),
                    #                                             axis=0)  # !!!track_bboxes2_func = !!!
                    #         matched_ids_func.append(A_new_ID_func[ii])
                    #         flag = 1
                    #         print("flag == 1")
                    #         cv2.rectangle(image2, (int(xmin2), int(ymin2)), (int(xmax2), int(ymax2)), (250, 33, 32), 5)
                    #         cv2.imshow("fksoadf", image2)
                    #         cv2.waitKey(100)
                    #         flag = 0
                    #         # print("new id", A_new_ID_func[ii])
                    #         # print(track_bboxes2_func[-1])

                        # 对于置信度高的可以直接映射过去
                        # if IOU_flag == 0:
                        #     print("directly_suppliment")
                        #     track_bboxes2_func = np.concatenate((track_bboxes2_func, np.array(
                        #         [[A_new_ID_func[ii], min_x, min_y, max_x, max_y, 0.999]])), axis=0)
                        #     matched_ids_func.append(A_new_ID_func[ii])
                        #     flag = 1
                        #     print("flag == 1")
                        #     cv2.rectangle(image2, (int(min_x), int(min_y)),
                        #                   (int(max_x), int(max_y)), (250, 33, 32), 5)
                        #     cv2.imshow("fksoadf", image2)
                        #     cv2.waitKey(1000)
                        #     flag = 0

    return track_bboxes_func, track_bboxes2_func, matched_ids_func, flag, coID_confirme