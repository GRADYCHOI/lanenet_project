#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math
import time

import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

np.set_printoptions(threshold=np.inf)

truck_end_line = 120
prev_left_x = 0
prev_left_y = 0
prev_right_x = 0
prev_right_y = 0

def _morphological_process(image, kernel_size=5): # to denoise binary line
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)
    # 5x5 ellipse matrix
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, cfg):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]
        self._cfg = cfg

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        db = DBSCAN(eps=self._cfg.POSTPROCESS.DBSCAN_EPS, min_samples=self._cfg.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        try:
            features = StandardScaler().fit_transform(embedding_image_feats) 
#            print(embedding_image_feats)
            #t_start = time.time()
            db.fit(features) # consume many time
#            print('db.fit(), cost time : {:.5f}s'.format(time.time() - t_start))
        except Exception as err:
            log.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

#        print(db_labels)
        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255) # binary_img's white fixel
        lane_embedding_feats = instance_seg_ret[idx] # fixel value
#        print(idx) #format:(arr(),arr())
        # idx_scale = np.vstack((idx[0] / 256.0, idx[1] / 512.0)).transpose()
        # lane_embedding_feats = np.hstack((lane_embedding_feats, idx_scale))
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose() # coordi

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result, truck_end):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        global prev_left_x, prev_right_x, prev_left_y, prev_right_y
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        #print(get_lane_embedding_feats_result['lane_embedding_feats'])
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None
        lane_coords = []
        lane_dec = []
        truck_end = (truck_end*416)//720
        print("new image")
        # draw line cluster 
        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue       #label = -1 --> pass
            idx = np.where(db_labels == label)   # db label == unique label -> pixel choice  
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            lane_cen_x = (coord[idx][:,0].sum())//len(coord[idx][:,0])  # lane average pixel make
            lane_cen_y = (coord[idx][:,1].sum())//len(coord[idx][:,1])
#            print("------------------")
#            print(len(coord[idx][:,0]))
            self_dist = math.sqrt(math.pow((lane_cen_x - 256), 2) + math.pow((lane_cen_y - 128), 2))
#            print(self_dist)
            lane_dec.append(label)
            if self_dist <= 140 and len(coord[idx][:,0]) >= 1300:   #find left, right lane
                #mask[pix_coord_idx] = self._color_map[1] 
                if max(coord[idx][:,0]) < 240:                #find left lane
                    left_xmin = min(coord[idx][:,0])
                    if (prev_left_x > 0) and (abs(prev_left_x - left_xmin) > 50):
                        left_xmin = prev_left_x
                        print("left! x")
                    left_ymax = max(coord[idx][:,1])
                    if (prev_left_y > 0) and (abs(prev_left_y - left_xmin) > 40):
                        left_xmin = prev_left_y
                        print("left! y")
                    left_xmax = max(coord[idx][:,0])
                    left_ymin = min(coord[idx][:,1])
                    #left_ymin = (truck_end*416)//720
                    prev_left_x = left_xmin
                    prev_left_y = left_xmin

                elif max(coord[idx][:,0]) >=240:              #find right lane
                    right_xmin = min(coord[idx][:,0])
                    right_ymin = min(coord[idx][:,1])
                    #right_ymin = (truck_end*416)//720
                    right_xmax = max(coord[idx][:,0])
                    if (prev_right_x > 0) and (abs(prev_right_x - right_xmax) > 50):
                        right_xmax = prev_right_x
                        print("right x!")
                    right_ymax = max(coord[idx][:,1])
                    if (prev_right_y > 0) and (abs(prev_right_y - right_xmax) > 40):
                        right_xmax = prev_right_y
                        print("right y!")
                    prev_right_x = right_xmax
                    prev_right_y = right_xmax
#            else :
#                mask[pix_coord_idx] = self._color_map[2] 


            #mask[pix_coord_idx] = self._color_map[index] 
            lane_coords.append(coord[idx])
        pts_all = np.array([[left_xmin,left_ymax], [left_xmin, 255], [left_xmax,left_ymin],[right_xmin,right_ymin], [right_xmax, 255], [right_xmax,right_ymax]])
        cv2.fillConvexPoly(mask, pts_all, (50,100,50))
        black_color = (0,0,0)
        cv2.rectangle(mask,(50,0),(500,truck_end), (0,0,0), -1)
#        cv2.imshow("mask", mask)



        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
#    def __init__(self, cfg, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):
    def __init__(self, cfg, ipm_remap_file_path='/home/choiin/lanenet_ws/src/lanenet_ros/scripts/data/tusimple_ipm_remap.yml'):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cfg = cfg
        self._cluster = _LaneNetCluster(cfg=cfg)
        self._ipm_remap_file_path = ipm_remap_file_path

        remap_file_load_ret = self._load_remap_matrix()
        self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

        self._color_map = [np.array([255, 0, 0]), # 8 color
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])] # each lane color??

    def _load_remap_matrix(self):
        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result=None, min_area_threshold=100, source_image=None, sub_truck_end=None, data_source='tusimple'):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param sub_truck_end:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

#        # zero vanishing point
#        binary_seg_result = cv2.circle(binary_seg_result, (230,40), 30, 0, -1)
#        dd


        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=4) # to denoise binary image

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)
        # c_c_a_r[0]:retval,the num of obj, [1]:labels, [2]:stats Nx5 mat, [3]:centroids
        labels = connect_components_analysis_ret[1]
#        print(type(labels))
#        print("labels = ", labels) 
        stats = connect_components_analysis_ret[2]
#        print(type(stats))
#        print("stats = ", stats)
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold: # area : stats[x,y,w,h,area] have 5 elements
                idx = np.where(labels == index)
                morphological_ret[idx] = 0  # some task to morphological_ret
        #cv2.imshow("morph", morphological_ret)

        white = (255,255,255)
        cir_img = np.zeros([256,512])
        cir_img = cv2.circle(cir_img, (256,10), 200, 255, 2)

        w_img = morphological_ret + cir_img
        w_img = np.where(w_img > 256, w_img, 0)



#        cv2.imshow("post_bin_img",morphological_ret)
#        cv2.imshow("post_cir_img",cir_img)
#        cv2.imshow("post_w_img",w_img)
        cv2.waitKey(1)



        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result,
            truck_end=sub_truck_end
        )
#        print(type(lane_coords))
#        print(type(mask_image))
#        cv2.imshow("mask", mask_image)

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
            }

        # lane line fit
#        t_start = time.time()
        fit_params = []
        src_lane_pts = []  # lane pts every single lane
        for lane_index, coords in enumerate(lane_coords):
            if data_source == 'tusimple':
                tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255
#                tmp_mask = np.zeros(shape=(480, 640), dtype=np.uint8)
#                tmp_mask[tuple((np.int_(coords[:, 1] * 480 / 256), np.int_(coords[:, 0] * 640 / 512)))] = 255
            else:
                raise ValueError('Wrong data source now only support tusimple')
            tmp_ipm_mask = cv2.remap(
                tmp_mask,
                self._remap_to_ipm_x,
                self._remap_to_ipm_y,
                interpolation=cv2.INTER_NEAREST
            )
            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])
#            print(nonzero_y)
#            print(nonzero_x)

            fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
#            print(len(plot_y))
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

            lane_pts = []
            for index in range(0, plot_y.shape[0], 5):
                src_x = self._remap_to_ipm_x[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                if src_x <= 0:
                    continue
                src_y = self._remap_to_ipm_y[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                src_y = src_y if src_y > 0 else 0

                lane_pts.append([src_x, src_y])

            src_lane_pts.append(lane_pts) # class = list
#        print("fit_params type = ", type(fit_params))
#        print(np.shape(fit_params))
        # tusimple test data sample point along y axis every 10 pixels
        source_image_width = source_image.shape[1]
#        print(src_lane_pts) # all lane points, class = list
#        print('postprocess 1, cost tiome : {:.5f}s'.format(time.time() - t_start))
#        t_start = time.time()
        line_count = 0
        for index, single_lane_pts in enumerate(src_lane_pts):
            line_count += 1
            #print("index = ", index)
            #print("single_lane_pts = ", single_lane_pts)
            single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
            single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
            if data_source == 'tusimple':
                start_plot_y = 0 #240
                end_plot_y = 720 #480
            else:
                raise ValueError('Wrong data source now only support tusimple')
            step = int(math.floor((end_plot_y - start_plot_y) / 20)) #10
            for plot_y in np.linspace(start_plot_y, end_plot_y, step):
                diff = single_lane_pt_y - plot_y
                fake_diff_bigger_than_zero = diff.copy()
                fake_diff_smaller_than_zero = diff.copy()
                fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
                fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
                idx_low = np.argmax(fake_diff_smaller_than_zero)
                idx_high = np.argmin(fake_diff_bigger_than_zero)

                previous_src_pt_x = single_lane_pt_x[idx_low]
                previous_src_pt_y = single_lane_pt_y[idx_low]
                last_src_pt_x = single_lane_pt_x[idx_high]
                last_src_pt_y = single_lane_pt_y[idx_high]
#                print('\nlast_src_pt_xy')
#                print(last_src_pt_x)
#                print(last_src_pt_y)

                if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                        fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                        fake_diff_bigger_than_zero[idx_high] == float('inf'):
                    continue

                interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

                if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 5: #10:
                    continue

                lane_color = self._color_map[index].tolist()
                cv2.circle(source_image, (int(interpolation_src_pt_x), int(interpolation_src_pt_y)), 5, lane_color, -1)

#        cv2.imshow("source", source_image)
#        print(type(source_image))

#        print('postprocess 2, cost tiome : {:.5f}s'.format(time.time() - t_start))
        ret = {
            'mask_image': mask_image,
            'line_count': line_count,
            'fit_params': fit_params,
            'source_image': source_image,
        }

        return ret
#        return 0
