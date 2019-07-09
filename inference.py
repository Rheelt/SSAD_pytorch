import os
import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from config import Config
from dataset import THUMOSInferenceDataset, inference_collate_fn
from model import SSAD
from utils import post_process, temporal_nms

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type('torch.FloatTensor')


def inference(config):
    # setup data_loader instances
    inference_loader = torch.utils.data.DataLoader(THUMOSInferenceDataset(config),
                                                   batch_size=config.batch_size, shuffle=False,
                                                   num_workers=8, pin_memory=True, drop_last=False,
                                                   collate_fn=inference_collate_fn)

    # build model architecture and load checkpoint
    model = SSAD(config).to(device)
    checkpoint = torch.load(config.checkpoint_path + "/model_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    '''
    ['xmin', 'xmax', 'conf', 'score_0', 'score_1', 'score_2',
                              'score_3', 'score_4', 'score_5', 'score_6', 'score_7', 'score_8',
                              'score_9', 'score_10', 'score_11', 'score_12', 'score_13', 'score_14',
                              'score_15', 'score_16', 'score_17', 'score_18', 'score_19', 'score_20']
    '''
    results = []
    results_name = []
    with torch.no_grad():
        for n_iter, (batch_data, batch_video_names, batch_window_start) in enumerate(inference_loader):
            batch_data = batch_data.to(device)
            output_x, output_w, output_scores, output_labels = model(batch_data, device)

            output_labels = F.softmax(output_labels, dim=1)
            output_x = output_x.cpu().detach().numpy()
            output_w = output_w.cpu().detach().numpy()
            output_scores = output_scores.cpu().detach().numpy()
            output_labels = output_labels.cpu().detach().numpy()
            output_min = output_x - output_w / 2
            output_max = output_x + output_w / 2
            for ii in range(len(batch_video_names)):
                video_name = batch_video_names[ii]
                window_start = batch_window_start[ii]
                a_min = output_min[ii, :]
                a_max = output_max[ii, :]
                a_scores = output_scores[ii, :]
                a_labels = output_labels[ii, :, :]
                for jj in range(output_min.shape[-1]):
                    corrected_min = max(a_min[jj] * config.window_size * config.unit_size, 0.) + window_start
                    corrected_max = min(a_max[jj] * config.window_size * config.unit_size,
                                        config.window_size * config.unit_size) + window_start
                    results_name.append([video_name])
                    results.append([corrected_min, corrected_max, a_scores[jj]] + a_labels[:, jj].tolist())
    results_name = np.stack(results_name)
    results = np.stack(results)
    df = pd.DataFrame(results, columns=config.outdf_columns)
    df['video_name'] = results_name
    result_file = './results.txt'
    if os.path.isfile(result_file):
        os.remove(result_file)
    df = df[df.score_0 < config.filter_neg_threshold]
    df = df[df.conf > config.filter_conf_threshold]
    video_name_list = list(set(df.video_name.values[:]))

    for video_name in video_name_list:
        tmpdf = df[df.video_name == video_name]
        tmpdf = post_process(tmpdf, config)

        temporal_nms(config, tmpdf, result_file, video_name)


if __name__ == '__main__':
    config = Config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    inference(config)
