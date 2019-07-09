import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from utils import ioa_with_anchors
from tqdm import tqdm
from config import Config


class THUMOSDataset(data.Dataset):

    def __init__(self, config, mode='Val'):
        self.feature_path = config.feature_path
        self.unit_size = config.unit_size
        self.feature_dim = config.feature_dim
        self.ioa_ratio_threshold = config.ioa_ratio_threshold
        self.window_size = config.window_size
        self.window_step = config.window_step
        self.num_classes = config.num_classes  # action categroies + BG for THUMOS14 is 21
        self.mode = mode
        self.anno_df = pd.read_csv("./data/thumos_14_annotations/" + mode + "_Annotation.csv")
        self.videoNameList = list(set(self.anno_df.video.values[:]))
        self.sampels = []
        self.class_real = [0] + [7, 9, 12, 21, 22, 23, 24, 26, 31, 33,
                                 36, 40, 45, 51, 68, 79, 85, 92, 93, 97]  # THUMOS14 calss label idx
        self._preparedata()
        print(
            'The number of {} dataset video is {} and the number of samples is {}'.format(mode, len(self.videoNameList),
                                                                                          len(self.sampels)))

    def _preparedata(self):
        print('wait...prepare data')
        for videoName in tqdm(self.videoNameList):
            video_annoDf = self.anno_df[self.anno_df.video == videoName]
            video_annoDf = video_annoDf[video_annoDf.type_idx != 0]  # 0 for Ambiguous

            gt_xmins = video_annoDf.startFrame.values[:]
            gt_xmaxs = video_annoDf.endFrame.values[:]
            gt_type_idx = video_annoDf.type_idx.values[:]

            rgb_feature, flow_feature = self._getVideoFeature(videoName, self.mode.lower())

            numSnippet = min(rgb_feature.shape[0], flow_feature.shape[0])
            frameList = [1 + self.unit_size * i for i in range(numSnippet)]
            df_data = np.concatenate((rgb_feature, flow_feature), axis=1)
            df_snippet = frameList
            window_size = self.window_size
            stride = self.window_step
            n_window = (numSnippet + stride - window_size) / stride
            windows_start = [i * stride for i in range(int(n_window))]
            if numSnippet < window_size:
                windows_start = [0]
                tmp_data = np.zeros((window_size - numSnippet, self.feature_dim))
                df_data = np.concatenate((df_data, tmp_data), axis=0)
                df_snippet.extend([df_snippet[-1] + self.unit_size * (i + 1) for i in range(window_size - numSnippet)])
            elif numSnippet - windows_start[-1] - window_size > 30:
                windows_start.append(numSnippet - window_size)

            snippet_xmin = df_snippet
            snippet_xmax = df_snippet[1:]
            snippet_xmax.append(df_snippet[-1] + self.unit_size)
            for start in windows_start:
                tmp_data = df_data[start:start + window_size, :]
                tmp_anchor_xmins = snippet_xmin[start:start + window_size]
                tmp_anchor_xmaxs = snippet_xmax[start:start + window_size]
                tmp_gt_bbox = []
                tmp_gt_class = []
                tmp_ioa_list = []
                for idx in range(len(gt_xmins)):
                    tmp_ioa = ioa_with_anchors(gt_xmins[idx], gt_xmaxs[idx], tmp_anchor_xmins[0], tmp_anchor_xmaxs[-1])
                    tmp_ioa_list.append(tmp_ioa)
                    if tmp_ioa > 0:
                        # gt bbox info
                        corrected_start = max(gt_xmins[idx], tmp_anchor_xmins[0]) - tmp_anchor_xmins[0]
                        corrected_end = min(gt_xmaxs[idx], tmp_anchor_xmaxs[-1]) - tmp_anchor_xmins[0]
                        tmp_gt_bbox.append([float(corrected_start) / (self.window_size * self.unit_size),
                                            float(corrected_end) / (self.window_size * self.unit_size)])
                        # gt class label
                        one_hot = [0] * self.num_classes
                        one_hot[self.class_real.index(gt_type_idx[idx])] = 1
                        tmp_gt_class.append(one_hot)
                if len(tmp_gt_bbox) > 0 and max(tmp_ioa_list) > self.ioa_ratio_threshold:
                    # the overlap region is corrected
                    tmp_results = [torch.transpose(torch.Tensor(tmp_data), 0, 1), np.array(tmp_gt_bbox),
                                   np.array(tmp_gt_class)]
                    self.sampels.append(tmp_results)

    def _getVideoFeature(self, videoname, subset):
        appearance_path = '~/THUMOS14_ANET_feature/{}_appearance/'.format(subset)
        denseflow_path = '~/THUMOS14_ANET_feature/{}_denseflow/'.format(subset)
        rgb_feature = np.load(appearance_path + videoname + '.npy')
        flow_feature = np.load(denseflow_path + videoname + '.npy')

        return rgb_feature, flow_feature

    def __getitem__(self, index):
        return self.sampels[index]

    def __len__(self):
        return len(self.sampels)


def train_collate_fn(batch):
    batch_start_index = [0]
    batch_gt_bbox = []
    batch_gt_class = []
    for iitem in batch:
        batch_start_index.append(batch_start_index[-1] + iitem[1].shape[0])
        batch_gt_bbox.append(iitem[1])
        batch_gt_class.append(iitem[2])
    batch_start_index = np.array(batch_start_index, dtype=np.int32)
    batch_data = torch.cat([x[0].unsqueeze(0) for x in batch])
    batch_gt_bbox = np.vstack(batch_gt_bbox).astype(np.float32)
    batch_gt_class = np.vstack(batch_gt_class).astype(np.int32)

    return batch_data, batch_gt_bbox, batch_gt_class, batch_start_index


class THUMOSInferenceDataset(data.Dataset):

    def __init__(self, config):
        self.feature_path = config.feature_path
        self.unit_size = config.unit_size
        self.feature_dim = config.feature_dim
        self.window_size = config.window_size
        self.inference_window_step = config.inference_window_step
        self.mode = 'Test'
        self.anno_df = pd.read_csv("./data/thumos_14_annotations/" + self.mode + "_Annotation.csv")
        self.videoNameList = list(set(self.anno_df.video.values[:]))
        self.sampels = []
        self._preparedata()
        print(
            'The number of {} dataset video is {} and the number of samples is {}'.format(self.mode,
                                                                                          len(self.videoNameList),
                                                                                          len(self.sampels)))

    def _preparedata(self):
        print('wait...prepare data')
        for videoName in tqdm(self.videoNameList):
            rgb_feature, flow_feature = self._getVideoFeature(videoName, self.mode.lower())

            numSnippet = min(rgb_feature.shape[0], flow_feature.shape[0])
            frameList = [1 + self.unit_size * i for i in range(numSnippet)]
            df_data = np.concatenate((rgb_feature, flow_feature), axis=1)
            df_snippet = frameList
            window_size = self.window_size
            stride = self.inference_window_step
            n_window = (numSnippet + stride - window_size) / stride
            windows_start = [i * stride for i in range(int(n_window))]
            if numSnippet < window_size:
                windows_start = [0]
                tmp_data = np.zeros((window_size - numSnippet, self.feature_dim))
                df_data = np.concatenate((df_data, tmp_data), axis=0)
                df_snippet.extend([df_snippet[-1] + self.unit_size * (i + 1) for i in range(window_size - numSnippet)])
            else:
                windows_start.append(numSnippet - window_size)

            snippet_xmin = df_snippet
            for start in windows_start:
                tmp_data = df_data[start:start + window_size, :]
                tmp_anchor_xmins = snippet_xmin[start:start + window_size]
                tmp_results = [torch.transpose(torch.Tensor(tmp_data), 0, 1), videoName, tmp_anchor_xmins[0]]
                self.sampels.append(tmp_results)

    def _getVideoFeature(self, videoname, subset):
        appearance_path = '~/THUMOS14_ANET_feature/{}_appearance/'.format(subset)
        denseflow_path = '~/THUMOS14_ANET_feature/{}_denseflow/'.format(subset)
        rgb_feature = np.load(appearance_path + videoname + '.npy')
        flow_feature = np.load(denseflow_path + videoname + '.npy')

        return rgb_feature, flow_feature

    def __getitem__(self, index):
        return self.sampels[index]

    def __len__(self):
        return len(self.sampels)


def inference_collate_fn(batch):
    batch_data = torch.cat([x[0].unsqueeze(0) for x in batch])
    batch_video_names = [x[1] for x in batch]
    batch_window_start = [x[2] for x in batch]
    return batch_data, batch_video_names, batch_window_start


if __name__ == '__main__':

    config = Config()
    train_loader = torch.utils.data.DataLoader(THUMOSInferenceDataset(config),
                                               batch_size=48, shuffle=False,
                                               num_workers=8, pin_memory=True, drop_last=False,
                                               collate_fn=inference_collate_fn)
    for idx, (batch_data, batch_video_names, batch_window_start) in enumerate(train_loader):
        print(idx)
        print(batch_data.shape[0])
