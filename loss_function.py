import torch
import torch.nn.functional as F


def SSAD_loss_function(all_prediction_x, all_prediction_w, all_prediction_score, all_prediction_label,
                       batch_match_x, batch_match_w, batch_match_scores, batch_match_labels, device, config):
    # calc Loss
    pmask = torch.ge(batch_match_scores, 0.5).float()
    num_positive = torch.sum(pmask)
    # print('num_positive', num_positive)
    num_entries = all_prediction_x.shape[0] * all_prediction_x.shape[1]

    hmask = batch_match_scores < 0.5
    hmask = hmask & (all_prediction_score > 0.5)
    hmask = hmask.float()
    num_hard = torch.sum(hmask)

    r_negative = (config.negative_ratio - num_hard / num_positive) * num_positive / (
            num_entries - num_positive - num_hard)
    r_negative = torch.min(r_negative, torch.Tensor([1.0]).to(device))
    nmask = torch.rand(pmask.size()).to(device)
    nmask = nmask * (1. - pmask)
    nmask = nmask * (1. - hmask)
    nmask = torch.ge(nmask, 1. - r_negative).float()
    # print(r_negative, num_positive, num_hard, torch.sum(nmask))
    # class_loss
    weights = pmask + nmask + hmask
    all_prediction_label = all_prediction_label.transpose(1, 2).contiguous().view(-1, config.num_classes)
    batch_match_labels = batch_match_labels.view(-1)
    class_loss = F.cross_entropy(all_prediction_label, batch_match_labels, reduction='none')
    class_loss = torch.sum(class_loss * weights.view(-1)) / torch.sum(weights)
    # loc_loss
    weights = pmask
    tmp_anchors_xmin = all_prediction_x - all_prediction_w / 2
    tmp_anchors_xmax = all_prediction_x + all_prediction_w / 2
    tmp_match_xmin = batch_match_x - batch_match_w / 2
    tmp_match_xmax = batch_match_x + batch_match_w / 2

    loc_loss = F.smooth_l1_loss(tmp_anchors_xmin, tmp_match_xmin, reduction='none') + F.smooth_l1_loss(
        tmp_anchors_xmax, tmp_match_xmax, reduction='none')
    loc_loss = torch.sum(loc_loss * weights) / torch.sum(weights)

    # conf loss
    weights = pmask + nmask + hmask
    # match_scores is from jaccard_with_anchors
    conf_loss = F.smooth_l1_loss(all_prediction_score, batch_match_scores, reduction='none')
    conf_loss = torch.sum(conf_loss * weights) / torch.sum(weights)

    loss = class_loss + 10. * loc_loss + 10. * conf_loss
    loss_dict = {"cost": loss, "class_loss": class_loss,
                 "loc_loss": loc_loss, "overlap_loss": conf_loss}
    return loss_dict
