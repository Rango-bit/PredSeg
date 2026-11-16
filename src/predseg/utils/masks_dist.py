import torch


def compute_mask_distance(
    masks_pred,
    masks_enc,
    grid_size=14,
    offset_context_loss=False,  # distance scale normalization for different patch grids
):

    distances = []

    enc_ids = masks_enc[0]  # [B, N_enc]

    # encoder positions
    enc_h = enc_ids // grid_size
    enc_w = enc_ids % grid_size
    enc_pos = torch.stack([enc_h, enc_w], dim=-1).float()

    for pred_ids in masks_pred:
        pred_h = pred_ids // grid_size
        pred_w = pred_ids % grid_size
        pred_pos = torch.stack([pred_h, pred_w], dim=-1).float()

        dist = torch.cdist(enc_pos, pred_pos, p=2)

        dmin = dist.min(dim=-1)[0]

        if offset_context_loss:
            dmin = dmin / (grid_size / 14.0)

        distances.append(dmin)

    return torch.cat(distances, dim=0)
