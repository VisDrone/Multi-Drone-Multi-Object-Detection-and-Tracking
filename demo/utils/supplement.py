# _*_ coding: utf-8 _*_
# @Time    :2022/12/7 16:38
# @Author  :LiuZhihao
# @File    :supplement.py

import numpy as np
import cv2

def not_matched_supplement(A_new_ID_func, A_pts_func, A_pts_corner_func, f1_func, cent_allclass2_func,
                                track_bboxes_func, track_bboxes2_func, matched_ids_func, det_bboxes, det_bboxes2,
                                image2, coID_confirme, supplement_bbox_func, thres=100):
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
                if min_x > 1920 or max_x < 0 or min_y > 1080 or max_y < 0:  # or min_x < 0 or min_y < 0 or max_x > 1920 or max_y > 1080:
                    continue
                else:
                    # for j, dots in enumerate(cent_allclass2_func):  # track_bboxes2  ndarray n*6
                    #     centx = int(dots[0])
                    #     centy = int(dots[1])
                    #     # cv2.circle(img2, (centx, centy), 9, (150, 34, 23), 3)
                    #     # cv2.circle(img2, (int(xy[0, 0]), int(xy[0, 1])), 9, (50, 340, 23), 3)
                    #
                    #     dist[ii, j] = ((xy[0, 0] - centx) ** 2 + (xy[0, 1] - centy) ** 2) ** 0.5
                    #     # cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)), (33, 33, 32), 5)
                    # # 超参数！35
                    # # print(ii, dist)
                    # if min(dist[ii], default=0) < thres:
                    #     # A:A_index   B:B_index
                    #     # A_index = list(track_bboxes[:, 0]).index(A_new_ID[ii])
                    #     # B_index = list(dist[ii]).index(min(dist[ii], default=0))
                    #     # print(np.where(track_bboxes_func[:, 0] == A_new_ID_func[ii])[0])
                    #     A_index = np.where(track_bboxes_func[:, 0] == A_new_ID_func[ii])[0]  # [0].astype(int)
                    #     B_index = np.where(dist[ii] == min(dist[ii]))[0]
                    #     if len(A_index) > 1:
                    #         A_index = min(
                    #             A_index)  # #####################??????????????????????????????????????????min???
                    #     else:
                    #         A_index = A_index[0]
                    #     if len(B_index) > 1:
                    #         B_index = min(
                    #             B_index)  # #####################??????????????????????????????????????????min???
                    #     else:
                    #         B_index = B_index[0]
                    #     # if track_bboxes2_func[B_index, 0] in matched_ids_func:  # 比较大小取小
                    #
                    #     if track_bboxes2_func[B_index, 0] not in matched_ids_func:
                    #         # #################matchede id里面还可能有错误匹配，进行进一步激励计算取最近的加入matched id,另一个重新加入unmatched ids，在后续进行rematch
                    #         # #################matchede id里面还可能有错误匹配，进行进一步激励计算取最近的加入matched id,另一个重新加入unmatched ids，在后续进行rematch
                    #         # #################matchede id里面还可能有错误匹配，进行进一步激励计算取最近的加入matched id,另一个重新加入unmatched ids，在后续进行rematch
                    #         # max_id = max(track_bboxes2[B_index, 0], A_new_ID[ii])
                    #         min_id = int(min(track_bboxes2_func[B_index, 0], A_new_ID_func[ii]))
                    #         track_bboxes_func[A_index, 0] = min_id
                    #         track_bboxes2_func[B_index, 0] = min_id
                    #         # 最重要的一行语句：
                    #         # print("coID confirme:", min_id)
                    #         print("step3: coID confirme: {0} {1} to {2}".format(track_bboxes_func[A_index, 0],
                    #                                                             track_bboxes2_func[B_index, 0], min_id))
                    #         coID_confirme.append(int(min_id))
                    #         matched_ids_func.append(int(min_id))

                    # else:
                    IOU_flag = 0
                    iou = np.array([])
                    # 让映射框和检测框做匹配，匹配上的检测框做为补充框并且加入matched ids
                    for boxx in det_bboxes2:
                        # 保证补充狂在图像内，防止溢出边界
                        min_x = min_x if min_x > 0 else 0
                        min_y = min_y if min_y > 0 else 0
                        max_x = max_x if max_x < 1920 else 1920
                        max_y = max_y if max_y < 1080 else 1080
                        # 下面是计算两个框IOU的程序：
                        xmin1, ymin1, xmax1, ymax1 = min_x, min_y, max_x, max_y
                        xmin2, ymin2, xmax2, ymax2 = boxx[0], boxx[1], boxx[2], boxx[3]

                        xx1 = np.max([xmin1, xmin2])
                        yy1 = np.max([ymin1, ymin2])
                        xx2 = np.min([xmax1, xmax2])
                        yy2 = np.min([ymax1, ymax2])

                        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                        inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                        iou = np.append(iou, inter_area / (area1 + area2 - inter_area + 1e-6))
                        # 如果两个框IOU大于一定阈值则进行检测框补充，否则直接进行映射狂补充
                    if max(iou) > 0.3:
                        # indexindex = np.where(iou == max(iou))[0].astype(int)
                        # boxx = det_bboxes2[indexindex][0]
                        # 选置信度高的
                        indexindex = np.where(iou > 0.3)

                        if len(indexindex[0]) == 1:
                            indexindex = np.where(iou == max(iou))[0].astype(int)
                            boxx = det_bboxes2[indexindex][0]
                        else:
                            # print(max(det_bboxes2[indexindex][:, -1]))
                            index = np.where(
                                np.array(det_bboxes2[indexindex][:, -1]) == max(det_bboxes2[indexindex][:, -1]))
                            # print(indexindex)
                            # print(index[0])
                            # print(indexindex[0][index[0]])
                            boxx = det_bboxes2[indexindex[0][index[0]]][0]
                        # print(boxx)
                        xmin2, ymin2, xmax2, ymax2 = boxx[0], boxx[1], boxx[2], boxx[3]
                        if (xmax2 - xmin2) < 20:
                            continue
                        if (ymax2 - ymin2) < 20:
                            continue
                        print("suppliment    if matched id is matched here, the final NMS will remove it!")
                        IOU_flag = 1
                        track_bboxes2_func = np.concatenate((track_bboxes2_func, np.array(
                            [[A_new_ID_func[ii], xmin2, ymin2, xmax2, ymax2, 0.999]])),
                                                            axis=0)  # !!!track_bboxes2_func = !!!
                        matched_ids_func.append(A_new_ID_func[ii])
                        coID_confirme.append(A_new_ID_func[ii])
                        flag = 1
                        # ****************************************************
                        if len(supplement_bbox_func)==0:
                            supplement_bbox_func = np.array([[A_new_ID_func[ii], xmin2, ymin2, xmax2, ymax2, 0.999]])
                        else:
                            supplement_bbox_func = np.concatenate(
                                (supplement_bbox_func, np.array([[A_new_ID_func[ii], xmin2, ymin2, xmax2, ymax2, 0.999]])),
                                axis=0)
                        # ****************************************************
                                
                                
                        # print("flag == 1")
                        # cv2.rectangle(image2, (int(xmin2), int(ymin2)), (int(xmax2), int(ymax2)), (250, 33, 32), 5)
                        # cv2.imshow("supplementA2B", image2)
                        # cv2.waitKey(100)
                        # flag = 0
                        # print("new id", A_new_ID_func[ii])
                        # print(track_bboxes2_func[-1])

                    # 对于置信度高的可以直接映射过去
                    # if IOU_flag == 0:
                    #     print("directly_suppliment")
                    #     track_bboxes2_func = np.concatenate((track_bboxes2_func, np.array(
                    #         [[A_new_ID_func[ii], min_x, min_y, max_x, max_y, 0.999]])), axis=0)
                    #     matched_ids_func.append(A_new_ID_func[ii])
                    #     coID_confirme.append(A_new_ID_func[ii])
                    #     flag = 1
                    #     print("flag == 1")
                    #     cv2.rectangle(image2, (int(min_x), int(min_y)),
                    #                   (int(max_x), int(max_y)), (250, 33, 32), 5)
                    #     cv2.imshow("directly supplement", image2)
                    #     cv2.waitKey(100)
                    #     flag = 0

    return track_bboxes_func, track_bboxes2_func, matched_ids_func, flag, coID_confirme, supplement_bbox_func


def low_confidence_target_refresh_same_ID(det_bboxes_func, det_bboxes2_func, track_bboxes_func, track_bboxes2_func,
                                          matched_ids_func, max_id_func,
                                          image1, image2, f1_func, track_bboxes_old, track_bboxes2_old):
    for bbox in det_bboxes_func:
        # print(bbox)
        xmin2, ymin2, xmax2, ymax2 = bbox[0], bbox[1], bbox[2], bbox[3]
        if bbox[-1] < 0.5:
            cv2.rectangle(image1, (int(xmin2), int(ymin2)), (int(xmax2), int(ymax2)), (250, 33, 32), 5)
            cv2.imshow("fksoadf2", image1)
            cv2.waitKey(10)
            pts = np.array([
                [xmin2, ymin2],
                [xmax2, ymax2]
            ]).reshape(-1, 1, 2).astype(np.float32)

            dst = cv2.perspectiveTransform(pts, f1_func)
            xl = min(dst[0][0][0], dst[1][0][0])
            yl = min(dst[0][0][1], dst[1][0][1])
            xr = max(dst[0][0][0], dst[1][0][0])
            yr = max(dst[0][0][1], dst[1][0][1])
            if xl > 1920 or xr < 0 or yl > 1080 or yr < 0:
                continue
            iou = np.array([])
            for bbox2 in det_bboxes2_func:
                # 保证补充狂在图像内，防止溢出边界
                xl = xl if xl > 0 else 0
                yl = yl if yl > 0 else 0
                xr = xr if xr < 1920 else 1920
                yr = yr if yr < 1080 else 1080
                # 下面是计算两个框IOU的程序：
                xmin1, ymin1, xmax1, ymax1 = xl, yl, xr, yr
                xmin2, ymin2, xmax2, ymax2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

                xx1 = np.max([xmin1, xmin2])
                yy1 = np.max([ymin1, ymin2])
                xx2 = np.min([xmax1, xmax2])
                yy2 = np.min([ymax1, ymax2])

                area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                iou = np.append(iou, inter_area / (area1 + area2 - inter_area + 1e-6))
            if max(iou) > 0.01:
                indexindex = np.where(iou > 0.01)

                if len(indexindex[0]) == 1:
                    indexindex = np.where(iou == max(iou))[0].astype(int)
                    bbox2 = det_bboxes2_func[indexindex][0]
                else:
                    # print(max([det_bboxes2[indexindex][0][-1]]))
                    index = np.where(
                        [det_bboxes2_func[indexindex][0][-1]] == max([det_bboxes2_func[indexindex][0][-1]]))
                    # print(indexindex)
                    # print(index[0])
                    # print(indexindex[0][index[0]])
                    bbox2 = det_bboxes2_func[indexindex[0][index[0]]][0]
                # cv2.rectangle(image2, (int(xl), int(yl)), (int(xr), int(yr)), (250, 33, 32), 5)
                cv2.rectangle(image2, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), (250, 33, 32), 5)
                cv2.imshow("fksoadf3", image2)
                cv2.waitKey(10)
                # 判断是否在追踪轨迹中，不在的话加入追踪轨迹
                flag_A = 0
                flag_B = 0
                iou = np.array([])
                for bbox3 in track_bboxes_func:
                    # 下面是计算两个框IOU的程序：
                    xmin1, ymin1, xmax1, ymax1 = bbox[0], bbox[1], bbox[2], bbox[3]
                    xmin2, ymin2, xmax2, ymax2 = bbox3[1], bbox3[2], bbox3[3], bbox3[4]

                    xx1 = np.max([xmin1, xmin2])
                    yy1 = np.max([ymin1, ymin2])
                    xx2 = np.min([xmax1, xmax2])
                    yy2 = np.min([ymax1, ymax2])

                    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                    iou = np.append(iou, inter_area / (area1 + area2 - inter_area + 1e-6))
                if max(iou) > 0.8:
                    flag_A = 1
                # 还需要确定另一张图像中该目标有没有已经被追到
                # 判断是否在追踪轨迹中，不在的话加入追踪轨迹
                iou = np.array([])
                for bbox4 in track_bboxes2_func:
                    # 下面是计算两个框IOU的程序：
                    xmin1, ymin1, xmax1, ymax1 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
                    xmin2, ymin2, xmax2, ymax2 = bbox4[1], bbox4[2], bbox4[3], bbox4[4]

                    xx1 = np.max([xmin1, xmin2])
                    yy1 = np.max([ymin1, ymin2])
                    xx2 = np.min([xmax1, xmax2])
                    yy2 = np.min([ymax1, ymax2])

                    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                    iou = np.append(iou, inter_area / (area1 + area2 - inter_area + 1e-6))
                if max(iou) > 0.8:
                    flag_B = 1
                if flag_A == flag_B == 0:

                    # 继续和上一帧追踪框比较，如有匹配上的，继承ID
                    if track_bboxes_old == []:
                        id = max_id_func + 1
                    else:
                        iou = np.array([])
                        for bbox4 in track_bboxes_old:
                            xmin1, ymin1, xmax1, ymax1 = bbox[0], bbox[1], bbox[2], bbox[3]
                            xmin2, ymin2, xmax2, ymax2 = bbox4[1], bbox4[2], bbox4[3], bbox4[4]
                            xx1 = np.max([xmin1, xmin2])
                            yy1 = np.max([ymin1, ymin2])
                            xx2 = np.min([xmax1, xmax2])
                            yy2 = np.min([ymax1, ymax2])

                            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                            area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                            inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                            iou = np.append(iou, inter_area / (area1 + area2 - inter_area + 1e-6))
                        if max(iou) > 0.01:
                            indexindex = np.where(iou == max(iou))[0].astype(int)
                            preID = track_bboxes_old[indexindex][0][0]
                            id = preID
                        else:
                            # 继续和上一帧追踪框比较，如有匹配上的，继承ID
                            iou = np.array([])
                            for bbox4 in track_bboxes2_old:
                                xmin1, ymin1, xmax1, ymax1 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
                                xmin2, ymin2, xmax2, ymax2 = bbox4[1], bbox4[2], bbox4[3], bbox4[4]
                                xx1 = np.max([xmin1, xmin2])
                                yy1 = np.max([ymin1, ymin2])
                                xx2 = np.min([xmax1, xmax2])
                                yy2 = np.min([ymax1, ymax2])

                                area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                                area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                                inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                                iou = np.append(iou, inter_area / (area1 + area2 - inter_area + 1e-6))
                            if max(iou) > 0.01:
                                indexindex = np.where(iou == max(iou))[0].astype(int)
                                preID = track_bboxes2_old[indexindex][0][0]
                                id = preID
                            else:
                                id = max_id_func + 1

                    cv2.rectangle(image2, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), (250, 33, 32),
                                  5)
                    cv2.putText(image2, str(id), (int(bbox2[0]), int(bbox2[3])), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (255, 0, 0), 2)
                    cv2.imshow("lowscore_supplimentB", image2)
                    cv2.waitKey(10)
                    cv2.rectangle(image1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (250, 33, 32),
                                  5)
                    cv2.putText(image1, str(id), (int(bbox[0]), int(bbox[3])), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (255, 0, 0), 2)
                    cv2.imshow("lowscore_supplimentA", image1)
                    cv2.waitKey(10)

                    print("lowscore_suppliment:")
                    track_bboxes_func = np.concatenate(
                        (track_bboxes_func, np.array([[id, bbox[0], bbox[1], bbox[2], bbox[3], 0.999]])),
                        axis=0)  # !!!track_bboxes2_func = !!!
                    track_bboxes2_func = np.concatenate(
                        (track_bboxes2_func, np.array([[id, bbox2[0], bbox2[1], bbox2[2], bbox2[3], 0.999]])),
                        axis=0)  # !!!track_bboxes2_func = !!!
                    max_id_func += 1

    return track_bboxes_func, track_bboxes2_func, matched_ids_func




