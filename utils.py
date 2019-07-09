import os
import numpy as np
import pandas as pd


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def sigmoid(X):
    # map [0,1] -> [0.5,0.73] (almost linearly) ([-1, 0] -> [0.26, 0.5])
    return 1.0 / (1.0 + np.exp(-1.0 * X))


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_taeget(all_prediction_x_np, all_prediction_w_np, batch_gt_bbox, batch_gt_class, batch_start_index, config):
    batch_match_x = []
    batch_match_w = []
    batch_match_scores = []
    batch_match_labels = []

    for idx in range(config.batch_size):
        b_anchors_rx = all_prediction_x_np[idx, ...]
        b_anchors_rw = all_prediction_w_np[idx, ...]

        b_gt_class = batch_gt_class[batch_start_index[idx]:batch_start_index[idx + 1], ...]
        b_gt_bbox = batch_gt_bbox[batch_start_index[idx]:batch_start_index[idx + 1], ...]
        assert b_gt_class.shape[0] == b_gt_bbox.shape[0]

        b_gt_num = b_gt_bbox.shape[0]
        num_all_anchors = all_prediction_x_np.shape[1]
        match_x = np.zeros((num_all_anchors), dtype=np.float32)
        match_w = np.zeros((num_all_anchors), dtype=np.float32)
        match_scores = np.zeros((num_all_anchors), dtype=np.float32)

        match_labels_other = np.ones((num_all_anchors, 1), dtype=np.int32)
        match_labels_class = np.zeros((num_all_anchors, config.num_classes - 1),
                                      dtype=np.int32)
        match_labels = np.hstack([match_labels_other, match_labels_class])

        for jj in range(b_gt_num):
            a_gt_min = b_gt_bbox[jj, 0]
            a_gt_max = b_gt_bbox[jj, 1]
            a_gt_class = b_gt_class[jj]
            # ground truth
            a_gt_x = (a_gt_max + a_gt_min) / 2
            a_gt_w = (a_gt_max - a_gt_min)

            # predict
            anchors_min = b_anchors_rx - b_anchors_rw / 2
            anchors_max = b_anchors_rx + b_anchors_rw / 2

            jaccards = iou_with_anchors(anchors_min, anchors_max, a_gt_min, a_gt_max)

            # jaccards > b_match_scores > -0.5 & jaccards > matching_threshold
            mask = jaccards > match_scores
            matching_threshold = 0.5
            mask = mask & (jaccards > matching_threshold)
            mask = mask & (match_scores > -0.5)

            imask = mask.astype(np.int32)
            fmask = mask.astype(np.float32)
            # Update values using mask.
            # if overlap enough, update b_match_* with gt, otherwise not update
            match_x = fmask * a_gt_x + (1 - fmask) * match_x
            match_w = fmask * a_gt_w + (1 - fmask) * match_w

            ref_label = np.zeros_like(match_labels, dtype=np.int32)
            ref_label = ref_label + a_gt_class

            match_labels = np.matmul(np.diag(imask), ref_label) + np.matmul(np.diag(1 - imask), match_labels)

            match_scores = np.maximum(jaccards, match_scores)

        batch_match_x.append(np.expand_dims(match_x, axis=0))
        batch_match_w.append(np.expand_dims(match_w, axis=0))
        batch_match_scores.append(np.expand_dims(match_scores, axis=0))
        batch_match_labels.append(np.expand_dims(match_labels, axis=0))
    batch_match_x = np.vstack(batch_match_x)
    batch_match_w = np.vstack(batch_match_w)
    batch_match_scores = np.vstack(batch_match_scores)
    batch_match_labels = np.vstack(batch_match_labels)
    batch_match_labels = np.argmax(batch_match_labels, axis=-1)
    return batch_match_x, batch_match_w, batch_match_scores, batch_match_labels


def post_process(df, config):
    class_scores_class = [(df['score_' + str(i)]).values[:].tolist() for i in range(21)]
    class_scores_seg = [[class_scores_class[j][i] for j in range(21)] for i in range(len(df))]

    class_real = [0] + config.class_real  # num_classes + 1

    # save the top 2 or 3 score element
    # append the largest score element
    class_type_list = []
    class_score_list = []
    for i in range(len(df)):
        class_score = np.array(class_scores_seg[i][1:]) * df.conf.values[i]
        class_score = class_score.tolist()
        class_type = class_real[class_score.index(max(class_score)) + 1]
        class_type_list.append(class_type)
        class_score_list.append(max(class_score))
    resultDf1 = pd.DataFrame()
    resultDf1['out_type'] = class_type_list
    resultDf1['out_score'] = class_score_list
    resultDf1['start'] = df.xmin.values[:]
    resultDf1['end'] = df.xmax.values[:]

    # append the second largest score element
    class_type_list = []
    class_score_list = []
    for i in range(len(df)):
        class_score = np.array(class_scores_seg[i][1:]) * df.conf.values[i]
        class_score = class_score.tolist()
        class_score[class_score.index(max(class_score))] = 0
        class_type = class_real[class_score.index(max(class_score)) + 1]
        class_type_list.append(class_type)
        class_score_list.append(max(class_score))
    resultDf2 = pd.DataFrame()
    resultDf2['out_type'] = class_type_list
    resultDf2['out_score'] = class_score_list
    resultDf2['start'] = df.xmin.values[:]
    resultDf2['end'] = df.xmax.values[:]
    resultDf1 = pd.concat([resultDf1, resultDf2])

    # append the third largest score element (improve little and slow)
    class_type_list = []
    class_score_list = []
    for i in range(len(df)):
        class_score = np.array(class_scores_seg[i][1:]) * df.conf.values[i]
        class_score = class_score.tolist()
        class_score[class_score.index(max(class_score))] = 0
        class_score[class_score.index(max(class_score))] = 0
        class_type = class_real[class_score.index(max(class_score)) + 1]
        class_type_list.append(class_type)
        class_score_list.append(max(class_score))
    resultDf2 = pd.DataFrame()
    resultDf2['out_type'] = class_type_list
    resultDf2['out_score'] = class_score_list
    resultDf2['start'] = df.xmin.values[:]
    resultDf2['end'] = df.xmax.values[:]
    resultDf1 = pd.concat([resultDf1, resultDf2])

    resultDf1 = resultDf1[resultDf1.out_score > 0.0005]

    resultDf1['video_name'] = [df['video_name'].values[0] for _ in range(len(resultDf1))]
    return resultDf1


def temporal_nms(config, dfNMS, filename, videoname):
    nms_threshold = config.nms_threshold
    fo = open(filename, 'a')

    typeSet = list(set(dfNMS.out_type.values[:]))
    for t in typeSet:
        tdf = dfNMS[dfNMS.out_type == t]

        t1 = np.array(tdf.start.values[:])
        t2 = np.array(tdf.end.values[:])
        scores = np.array(tdf.out_score.values[:])
        ttype = list(tdf.out_type.values[:])

        durations = t2 - t1
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            tt1 = np.maximum(t1[i], t1[order[1:]])
            tt2 = np.minimum(t2[i], t2[order[1:]])
            intersection = tt2 - tt1
            IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

            inds = np.where(IoU <= nms_threshold)[0]
            order = order[inds + 1]

        for idx in keep:
            # class_real: do not have class 0 (ambiguous) -> remove all ambiguous class
            if ttype[idx] in config.class_real:
                if videoname in ["video_test_0001255", "video_test_0001058",
                                 "video_test_0001459", "video_test_0001195", "video_test_0000950"]:  # 25fps
                    strout = "%s\t%.3f\t%.3f\t%d\t%.4f\n" % (videoname, float(t1[idx]) / 25,
                                                             float(t2[idx]) / 25, ttype[idx], scores[idx])
                elif videoname == "video_test_0001207":  # 24fps
                    strout = "%s\t%.3f\t%.3f\t%d\t%.4f\n" % (videoname, float(t1[idx]) / 24,
                                                             float(t2[idx]) / 24, ttype[idx], scores[idx])
                else:  # most videos are 30fps
                    strout = "%s\t%.3f\t%.3f\t%d\t%.4f\n" % (videoname, float(t1[idx]) / 30,
                                                             float(t2[idx]) / 30, ttype[idx], scores[idx])
                fo.write(strout)
