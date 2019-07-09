# -*- coding: utf-8 -*-
"""
Based on https://github.com/HYPJUDY

Single Shot Temporal Action Detection
-----------------------------------------------------------------------------------
SSAD
"""

from operations import *
from config import Config
import time
from os.path import join
import load_training_data as load_data_Train
import load_inference_data as load_data_Test

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

stage = 'train'  # train/test

unit_size = 5
feature_dim = 2048 + 1024

models_dir = './models/'
models_file_prefix = join(models_dir, 'model-ep')
test_checkpoint_file = join(models_dir, 'model-ep-34')


######################################### TRAIN ##########################################

def train_operation(X, Y_label, Y_bbox, Index, LR, config):
    bsz = config.batch_size
    ncls = config.num_classes

    net = base_feature_network(X)
    MALs = main_anchor_layer(net)

    # --------------------------- Main Stream -----------------------------
    full_mainAnc_class = tf.reshape(tf.constant([]), [bsz, -1, ncls])
    full_mainAnc_conf = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmin = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmax = tf.reshape(tf.constant([]), [bsz, -1])

    full_mainAnc_BM_x = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_BM_w = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_BM_labels = tf.reshape(tf.constant([], dtype=tf.int32), [bsz, -1, ncls])
    full_mainAnc_BM_scores = tf.reshape(tf.constant([]), [bsz, -1])

    for i, ln in enumerate(config.layers_name):
        mainAnc = mulClsReg_predict_layer(config, MALs[i], ln, 'mainStream')

        # --------------------------- Main Stream -----------------------------
        [mainAnc_BM_x, mainAnc_BM_w, mainAnc_BM_labels, mainAnc_BM_scores,
         mainAnc_class, mainAnc_conf, mainAnc_rx, mainAnc_rw] = \
            anchor_bboxes_encode(mainAnc, Y_label, Y_bbox, Index, config, ln)

        mainAnc_xmin = mainAnc_rx - mainAnc_rw / 2
        mainAnc_xmax = mainAnc_rx + mainAnc_rw / 2

        full_mainAnc_class = tf.concat([full_mainAnc_class, mainAnc_class], axis=1)
        full_mainAnc_conf = tf.concat([full_mainAnc_conf, mainAnc_conf], axis=1)
        full_mainAnc_xmin = tf.concat([full_mainAnc_xmin, mainAnc_xmin], axis=1)
        full_mainAnc_xmax = tf.concat([full_mainAnc_xmax, mainAnc_xmax], axis=1)

        full_mainAnc_BM_x = tf.concat([full_mainAnc_BM_x, mainAnc_BM_x], axis=1)
        full_mainAnc_BM_w = tf.concat([full_mainAnc_BM_w, mainAnc_BM_w], axis=1)
        full_mainAnc_BM_labels = tf.concat([full_mainAnc_BM_labels, mainAnc_BM_labels], axis=1)
        full_mainAnc_BM_scores = tf.concat([full_mainAnc_BM_scores, mainAnc_BM_scores], axis=1)

    main_class_loss, main_loc_loss, main_conf_loss, num_entries, num_positive, num_hard, num_easy = \
        loss_function(full_mainAnc_class, full_mainAnc_conf,
                      full_mainAnc_xmin, full_mainAnc_xmax,
                      full_mainAnc_BM_x, full_mainAnc_BM_w,
                      full_mainAnc_BM_labels, full_mainAnc_BM_scores, config)

    loss = main_class_loss + config.p_loc * main_loc_loss + config.p_conf * main_conf_loss

    trainable_variables = get_trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss, var_list=trainable_variables)

    return optimizer, loss, main_class_loss, main_loc_loss, main_conf_loss, num_entries, num_positive, num_hard, num_easy, trainable_variables


def train_main(config):
    bsz = config.batch_size

    tf.set_random_seed(config.seed)
    X = tf.placeholder(tf.float32, shape=(bsz, config.input_steps, feature_dim))
    Y_label = tf.placeholder(tf.int32, [None, config.num_classes])
    Y_bbox = tf.placeholder(tf.float32, [None, 2])
    Index = tf.placeholder(tf.int32, [bsz + 1])
    LR = tf.placeholder(tf.float32)

    optimizer, loss, main_class_loss, main_loc_loss, main_conf_loss, num_entries, num_positive, num_hard, num_easy, trainable_variables = \
        train_operation(X, Y_label, Y_bbox, Index, LR, config)

    model_saver = tf.train.Saver(var_list=trainable_variables, max_to_keep=2)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    sess = tf.InteractiveSession(config=tf_config)

    tf.global_variables_initializer().run()

    # initialize parameters or restore from previous model
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if os.listdir(models_dir) == [] or config.initialize:
        init_epoch = 0
        print("Initializing Network")
    else:
        init_epoch = int(config.steps)
        restore_checkpoint_file = join(models_dir, 'model-ep-' + str(config.steps - 1))
        model_saver.restore(sess, restore_checkpoint_file)

    trainDataDict = load_data_Train.getFullData("Val")
    for epoch in range(init_epoch, config.training_epochs):
        ## TRAIN ##
        batch_window_list = load_data_Train.getBatchList(len(trainDataDict["gt_bbox"]), config.batch_size,
                                                         shuffle=True)
        loss_info = []
        main_class_loss_info = []
        main_loc_loss_info = []
        main_conf_loss_info = []
        for idx in range(len(batch_window_list)):
            batch_index, batch_bbox, batch_label, batch_anchor_feature = load_data_Train.getBatchData(
                batch_window_list[idx], trainDataDict)
            feed_dict = {X: batch_anchor_feature,
                         Y_label: batch_label,
                         Y_bbox: batch_bbox,
                         Index: batch_index,
                         LR: config.learning_rates[epoch]}
            _, out_loss, out_main_class_loss, out_main_loc_loss, out_main_conf_loss, out_num_entries, out_num_positive, out_num_hard, out_num_easy = sess.run(
                [optimizer, loss, main_class_loss, main_loc_loss, main_conf_loss, num_entries, num_positive, num_hard,
                 num_easy, ],
                feed_dict=feed_dict)
            print out_num_entries, out_num_positive, out_num_hard, out_num_easy
            loss_info.append(out_loss)
            main_class_loss_info.append(out_main_class_loss)
            main_loc_loss_info.append(out_main_loc_loss)
            main_conf_loss_info.append(out_main_conf_loss)

        print (
            "Training epoch ", epoch, " loss: ", np.mean(loss_info), " main_class_loss: ",
            np.mean(main_class_loss_info),
            " main_loc_loss: ", np.mean(main_loc_loss_info),
            " main_conf_loss: ", np.mean(main_conf_loss_info))
        if epoch == config.training_epochs - 2 or epoch == config.training_epochs - 1:
            model_saver.save(sess, models_file_prefix, global_step=epoch)


########################################### TEST ############################################

def test_operation(X, config):
    bsz = config.batch_size
    ncls = config.num_classes

    net = base_feature_network(X)
    MALs = main_anchor_layer(net)

    full_mainAnc_class = tf.reshape(tf.constant([]), [bsz, -1, ncls])
    full_mainAnc_conf = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmin = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmax = tf.reshape(tf.constant([]), [bsz, -1])

    for i, ln in enumerate(config.layers_name):
        mainAnc = mulClsReg_predict_layer(config, MALs[i], ln, 'mainStream')

        mainAnc_class, mainAnc_conf, mainAnc_rx, mainAnc_rw = anchor_box_adjust(mainAnc, config, ln)

        mainAnc_xmin = mainAnc_rx - mainAnc_rw / 2
        mainAnc_xmax = mainAnc_rx + mainAnc_rw / 2

        full_mainAnc_class = tf.concat([full_mainAnc_class, mainAnc_class], axis=1)
        full_mainAnc_conf = tf.concat([full_mainAnc_conf, mainAnc_conf], axis=1)
        full_mainAnc_xmin = tf.concat([full_mainAnc_xmin, mainAnc_xmin], axis=1)
        full_mainAnc_xmax = tf.concat([full_mainAnc_xmax, mainAnc_xmax], axis=1)

    full_mainAnc_class = tf.nn.softmax(full_mainAnc_class, dim=-1)
    return full_mainAnc_class, full_mainAnc_conf, full_mainAnc_xmin, full_mainAnc_xmax


def test_main(config):
    X = tf.placeholder(tf.float32, shape=(config.batch_size, config.input_steps, feature_dim))

    anchors_class, anchors_conf, anchors_xmin, anchors_xmax = test_operation(X, config)

    model_saver = tf.train.Saver()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    sess = tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()
    model_saver.restore(sess, test_checkpoint_file)
    batch_winInfo = []

    batch_result_class = []
    batch_result_conf = []
    batch_result_xmin = []
    batch_result_xmax = []

    testDataDict = load_data_Test.getFullData("Test")
    batch_window_list = load_data_Test.getBatchList(len(testDataDict["info"]), config.batch_size, shuffle=False)
    num_batch = len(batch_window_list)
    for idx in range(len(batch_window_list)):
        batch_anchor_feature, batch_info = load_data_Test.getBatchData(
            batch_window_list[idx], testDataDict)
        batch_winInfo.append(batch_info)
        out_anchors_class, out_anchors_conf, out_anchors_xmin, out_anchors_xmax = \
            sess.run([anchors_class, anchors_conf, anchors_xmin, anchors_xmax],
                     feed_dict={X: batch_anchor_feature})

        batch_result_class.append(out_anchors_class)
        batch_result_conf.append(out_anchors_conf)
        batch_result_xmin.append(out_anchors_xmin * config.window_size)
        batch_result_xmax.append(out_anchors_xmax * config.window_size)

    outDf = pd.DataFrame(columns=config.outdf_columns)

    for i in range(num_batch):
        tmpDf = result_process(batch_winInfo, batch_result_class, batch_result_conf,
                               batch_result_xmin, batch_result_xmax, config, i)

        outDf = pd.concat([outDf, tmpDf])
    return outDf


if __name__ == "__main__":
    config = Config()
    start_time = time.time()
    if stage == 'train':
        train_main(config)
    elif stage == 'test':
        df = test_main(config)
        final_result_process(stage, config, df)
    else:
        print("No stage", stage, "Please choose a stage from train/test.")
