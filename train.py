import os
import torch
print(torch.__version__)
import torch.optim as optim
import random
import numpy as np
from config import Config
from dataset import THUMOSDataset, train_collate_fn
from model import SSAD
from utils import ensure_dir, build_taeget
from loss_function import SSAD_loss_function

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda')
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type('torch.FloatTensor')


def main(config):
    # setup data_loader instances
    train_loader = torch.utils.data.DataLoader(THUMOSDataset(config, mode='Val'),
                                               batch_size=config.batch_size, shuffle=True,
                                               num_workers=8, pin_memory=True, drop_last=True,
                                               collate_fn=train_collate_fn)
    val_loader = torch.utils.data.DataLoader(THUMOSDataset(config, mode='Test'),
                                             batch_size=config.batch_size, shuffle=False,
                                             num_workers=8, pin_memory=True, drop_last=True,
                                             collate_fn=train_collate_fn)

    # build model architecture
    model = SSAD(config).to(device)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=config.training_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_scheduler_step,
                                                gamma=config.lr_scheduler_gama)

    # Save configuration file into checkpoint directory:
    ensure_dir(config.checkpoint_path)

    for epoch in range(config.epoch):
        scheduler.step()
        train_epoch(train_loader, model, optimizer, epoch, config)
        test_epoch(val_loader, model, epoch, config)


def train_epoch(data_loader, model, optimizer, epoch, config):
    model.train()
    epoch_cost = 0.
    epoch_class_loss = 0.
    epoch_overlap_loss = 0.
    epoch_loc_loss = 0.
    for n_iter, (batch_data, batch_gt_bbox, batch_gt_class, batch_start_index) in enumerate(data_loader):
        batch_data = batch_data.to(device)

        all_prediction_x, all_prediction_w, all_prediction_score, all_prediction_label = model(batch_data, device)

        all_prediction_x_np = all_prediction_x.data.cpu().numpy()
        all_prediction_w_np = all_prediction_w.data.cpu().numpy()
        batch_match_x, batch_match_w, batch_match_scores, batch_match_labels = build_taeget(all_prediction_x_np,
                                                                                            all_prediction_w_np,
                                                                                            batch_gt_bbox,
                                                                                            batch_gt_class,
                                                                                            batch_start_index, config)
        batch_match_x = torch.Tensor(batch_match_x).to(device)
        batch_match_w = torch.Tensor(batch_match_w).to(device)
        batch_match_scores = torch.Tensor(batch_match_scores).to(device)
        batch_match_labels = torch.LongTensor(batch_match_labels).to(device)

        loss = SSAD_loss_function(all_prediction_x, all_prediction_w, all_prediction_score, all_prediction_label,
                                        batch_match_x, batch_match_w, batch_match_scores, batch_match_labels, device,
                                        config)
        cost = loss["cost"]

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        epoch_class_loss += loss["class_loss"].cpu().detach().numpy()
        epoch_overlap_loss += loss["overlap_loss"].cpu().detach().numpy()
        epoch_loc_loss += loss["loc_loss"].cpu().detach().numpy()
        epoch_cost += loss["cost"].cpu().detach().numpy()
    print(
        "SSAD training loss(epoch %d): class - %.05f, overlap - %.05f, loc - %.05f, cost - %.05f" % (
            epoch, epoch_class_loss / (n_iter + 1),
            epoch_overlap_loss / (n_iter + 1),
            epoch_loc_loss / (n_iter + 1), epoch_cost / (n_iter + 1)))


def test_epoch(data_loader, model, epoch, config):
    model.eval()
    epoch_cost = 0.
    epoch_class_loss = 0.
    epoch_overlap_loss = 0.
    epoch_loc_loss = 0.
    for n_iter, (batch_data, batch_gt_bbox, batch_gt_class, batch_start_index) in enumerate(data_loader):
        batch_data = batch_data.to(device)

        all_prediction_x, all_prediction_w, all_prediction_score, all_prediction_label = model(batch_data, device)

        all_prediction_x_np = all_prediction_x.data.cpu().numpy()
        all_prediction_w_np = all_prediction_w.data.cpu().numpy()
        batch_match_x, batch_match_w, batch_match_scores, batch_match_labels = build_taeget(all_prediction_x_np,
                                                                                            all_prediction_w_np,
                                                                                            batch_gt_bbox,
                                                                                            batch_gt_class,
                                                                                            batch_start_index, config)
        batch_match_x = torch.Tensor(batch_match_x).to(device)
        batch_match_w = torch.Tensor(batch_match_w).to(device)
        batch_match_scores = torch.Tensor(batch_match_scores).to(device)
        batch_match_labels = torch.LongTensor(batch_match_labels).to(device)

        loss = SSAD_loss_function(all_prediction_x, all_prediction_w, all_prediction_score, all_prediction_label,
                                        batch_match_x, batch_match_w, batch_match_scores, batch_match_labels, device,
                                        config)

        epoch_class_loss += loss["class_loss"].cpu().detach().numpy()
        epoch_overlap_loss += loss["overlap_loss"].cpu().detach().numpy()
        epoch_loc_loss += loss["loc_loss"].cpu().detach().numpy()
        epoch_cost += loss["cost"].cpu().detach().numpy()
    print(
        "SSAD validation loss(epoch %d): class - %.05f, overlap - %.05f, loc - %.05f, cost - %.05f" % (
            epoch, epoch_class_loss / (n_iter + 1),
            epoch_overlap_loss / (n_iter + 1),
            epoch_loc_loss / (n_iter + 1), epoch_cost / (n_iter + 1)))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, config.checkpoint_path + "/model_checkpoint.pth.tar")
    if np.mean(epoch_cost) < model.best_loss:
        model.best_loss = np.mean(epoch_cost)
        torch.save(state, config.checkpoint_path + "/model_best.pth.tar")


if __name__ == '__main__':
    config = Config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    main(config)
