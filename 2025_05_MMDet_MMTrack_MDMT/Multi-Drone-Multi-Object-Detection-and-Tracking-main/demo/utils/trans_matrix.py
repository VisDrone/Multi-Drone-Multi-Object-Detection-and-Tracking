# _*_ coding: utf-8 _*_
# @Time    :2022/12/7 16:38
# @Author  :LiuZhihao
# @File    :trans_matrix.py

import numpy as np
import cv2
from .matching_pure import matching, calculate_cent_corner_pst


def supp_compute_transf_matrix(pts_src, pts_dst, f_last, image11, image22):
    if len(pts_src) >= 5:
        # print("可计算旋转矩阵")
        f, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
        # print(f, f_last)
        if f is None:
            f = f_last.copy()
        f_last = f.copy()
        
        # if f is not None:  # 可能计算出来是None
        #     cosine_sim = f.reshape(1, -1).dot(f_last.reshape(1, -1).T)/(np.linalg.norm(f.reshape(1, -1)) * np.linalg.norm(f_last.reshape(1, -1)))
        #     print("cosine_sim", cosine_sim)
        #     if cosine_sim[0] > 0.99:  # 0.97就大不一样了
        #         f_last = f.copy()
        #     else:
        #         print('using global matching:')
        #         M = matching(image1, image2)
        #         cosine_sim = M.reshape(1, -1).dot(f_last.reshape(1, -1).T) / (
        #                 np.linalg.norm(M.reshape(1, -1)) * np.linalg.norm(f_last.reshape(1, -1)))
        #         print("cosine_sim", cosine_sim)
        #         if cosine_sim[0] > 0.99:  # 0.97就大不一样了
        #             f = ( 3 *f_last +  M) / 4
        #             f_last = f.copy()
        #         else:
        #             f = f_last.copy()
        # else:
        #     f = f_last.copy()
    else:
        M = matching(image11, image22)
        M2 = matching(image11, image22)
        M3 = matching(image11, image22)
        cosine_sim = M.reshape(1, -1).dot(f_last.reshape(1, -1).T) / (
                np.linalg.norm(M.reshape(1, -1)) * np.linalg.norm(f_last.reshape(1, -1)))
        print("cosine_sim", cosine_sim)
        if cosine_sim[0] > 0.99:  # 0.97就大不一样了
            print('using global matching:')
            f = M
            f_last = f.copy()
        else:
            print("global matching filtered:")
            cosine_sim12 = M.reshape(1, -1).dot(M2.reshape(1, -1).T) / (
                    np.linalg.norm(M.reshape(1, -1)) * np.linalg.norm(M2.reshape(1, -1)))
            cosine_sim13 = M.reshape(1, -1).dot(M3.reshape(1, -1).T) / (
                    np.linalg.norm(M.reshape(1, -1)) * np.linalg.norm(M3.reshape(1, -1)))
            cosine_sim23 = M2.reshape(1, -1).dot(M3.reshape(1, -1).T) / (
                    np.linalg.norm(M2.reshape(1, -1)) * np.linalg.norm(M3.reshape(1, -1)))
            if cosine_sim12[0] > 0.99:  # 0.97就大不一样了
                print("using filter12")
                f = (M + M2) / 2
                f_last = f.copy()
            elif cosine_sim13[0] > 0.99:  # 0.97就大不一样了
                print("using filter13")
                f = (M + M3) / 2
                f_last = f.copy()
            elif cosine_sim23[0] > 0.99:  # 0.97就大不一样了
                print("using filter23")
                f = (M2 + M3) / 2
                f_last = f.copy()
            else:
                print("using last transform matrix")
                f = f_last.copy()
    return f, f_last


def global_compute_transf_matrix(pts_src, pts_dst, f_last, image11, image22):

    M = matching(image11, image22)
    f = M
    
    if isinstance(f, int) or f.all() == 0:
        f_last = f
    else:
        f_last = f.copy()

    return f, f_last


def local_compute_transf_matrix(pts_src, pts_dst, f_last, image11, image22):
    
    # if len(pts_src) >= 5:
    # print("可计算旋转矩阵")
    f, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    f_last = f.copy()
    if f is None:
        f = f_last.copy()
        
    return f, f_last
