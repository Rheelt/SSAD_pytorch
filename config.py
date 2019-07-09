class Config(object):
    """
    define a class to store parameters,
    """

    def __init__(self):
        self.name = "SSAD"
        self.seed = 5
        self.feature_path = "~/THUMOS14_ANET_feature/"
        self.unit_size = 5
        self.feature_dim = 3072
        self.ioa_ratio_threshold = 0.9
        self.window_size = 128
        self.window_step = 64  # 75% overlap
        self.inference_window_step = 64  # 50% overlap
        self.num_classes = 21
        self.batch_size = 48

        # self.base_scale = {"AL1": 1. / 16, "AL2": 1. / 8, "AL3": 1. / 4}
        # self.num_cells = {"AL1": 16, "AL2": 8, "AL3": 4}
        # self.aspect_ratios = {"AL1": [0.5, 0.75, 1, 1.5, 2],
        #                       "AL2": [0.5, 0.75, 1, 1.5, 2],
        #                       "AL3": [0.5, 0.75, 1, 1.5, 2]}
        self.layer_names = ['AL1', 'AL2', 'AL3']
        self.base_scale = [1. / 16, 1. / 8, 1. / 4]
        self.num_cells = [16, 8, 4]
        self.aspect_ratios = [[0.5, 0.75, 1., 1.5, 2.],
                              [0.5, 0.75, 1., 1.5, 2.],
                              [0.5, 0.75, 1., 1.5, 2.]]

        self.num_anchors = 5
        self.training_lr = 0.0001
        self.weight_decay = 0.0
        self.checkpoint_path = "./checkpoint/"
        self.epoch = 35
        self.negative_ratio = 1.
        self.lr_scheduler_step = 30
        self.lr_scheduler_gama = 0.1

        self.outdf_columns = ['xmin', 'xmax', 'conf', 'score_0', 'score_1', 'score_2',
                              'score_3', 'score_4', 'score_5', 'score_6', 'score_7', 'score_8',
                              'score_9', 'score_10', 'score_11', 'score_12', 'score_13', 'score_14',
                              'score_15', 'score_16', 'score_17', 'score_18', 'score_19', 'score_20']
        self.class_real = [7, 9, 12, 21, 22, 23, 24, 26, 31, 33,
                           36, 40, 45, 51, 68, 79, 85, 92, 93, 97]
        self.nms_threshold = 0.2
        # when process results, remove confident negative anchors by previous
        self.filter_neg_threshold = 0.7
        # when process results, remove confident low overlap (conf) anchors by previous
        self.filter_conf_threshold = 0.3
