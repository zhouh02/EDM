from loguru import logger

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class FocalLossCovi(nn.Module):
    """
    Focal Loss for covisibility supervision.

    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    FL(p) = -alpha * (1 - p)^gamma * log(p) for positive
           -(1 - alpha) * p^gamma * log(1 - p) for negative
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1, H, W] - raw logits (before sigmoid)
            targets: [B, 1, H, W] - binary targets (0 or 1)
        """
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # [B, 1, H, W]

        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class EDMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.coord_length = config["edm"]["fine"]["coord_length"]

        self.loss_config = config["edm"]["loss"]
        self.sparse_spvs = self.loss_config["sparse_spvs"]

        # coarse-level
        self.c_pos_w = self.loss_config["pos_weight"]
        self.c_neg_w = self.loss_config["neg_weight"]
        # fine-level
        self.q_distribution = self.loss_config["q_distribution"]
        # covisibility
        self.covi_weight = self.loss_config["covi_weight"]
        self.covi_focal_alpha = self.loss_config.get("covi_focal_alpha", 0.25)
        self.covi_focal_gamma = self.loss_config.get("covi_focal_gamma", 2.0)
        self.covi_multi_stage_avg = self.loss_config.get("covi_multi_stage_avg", True)

        # DCAT covisibility loss
        self.dcat_enabled = config["edm"]["neck"]["dcat"]["enabled"]
        self.covi_focal_loss = FocalLossCovi(
            alpha=self.covi_focal_alpha,
            gamma=self.covi_focal_gamma,
            reduction="mean"
        )

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_pos_w = 0.0
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_neg_w = 0.0

        if self.loss_config["coarse_type"] == "cross_entropy":
            assert (
                not self.sparse_spvs
            ), "Sparse Supervision for cross-entropy not implemented!"
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = -torch.log(conf[pos_mask])
            loss_neg = -torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()

        elif self.loss_config["coarse_type"] == "focal":
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config["focal_alpha"]
            gamma = self.loss_config["focal_gamma"]

            if self.sparse_spvs:
                pos_conf = conf[pos_mask]
                loss_pos = -alpha * \
                    torch.pow(1 - pos_conf, gamma) * pos_conf.log()

                # handle loss weights
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]

                loss = c_pos_w * loss_pos.mean()
                return loss
            else:
                loss_pos = (
                    -alpha
                    * torch.pow(1 - conf[pos_mask], gamma)
                    * (conf[pos_mask]).log()
                )

                loss_neg = (
                    -alpha
                    * torch.pow(conf[neg_mask], gamma)
                    * (1 - conf[neg_mask]).log()
                )
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]

                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        else:
            raise ValueError(
                "Unknown coarse loss: {type}".format(
                    type=self.loss_config["coarse_type"]
                )
            )

    def compute_covi_loss_single(self, logit, gt):
        """
        Compute single-stage covisibility loss (focal loss).

        Args:
            logit: [B, 1, H, W] - predictor output
            gt: [B, 1, H, W] - covisibility ground truth
        """
        return self.covi_focal_loss(logit, gt)

    def compute_covi_loss_multi_stage(self, stage_logits, covi_gt_0, covi_gt_1):
        """
        Compute multi-stage covisibility loss (DCAT, CoMatch-aligned: 3 stages).

        All 3 stage predictor outputs are supervised with focal loss.
        Loss is averaged across stages and images.

        Args:
            stage_logits: list of 3 tuples (logit0, logit1) from cross i=1,3,5
            covi_gt_0: [B, 1, H0, W0]
            covi_gt_1: [B, 1, H1, W1]
        """
        if stage_logits is None or len(stage_logits) == 0:
            return None

        stage_losses = []
        for i, (logit0, logit1) in enumerate(stage_logits):
            # Loss for image 0
            loss_0 = self.compute_covi_loss_single(logit0, covi_gt_0)
            # Loss for image 1
            loss_1 = self.compute_covi_loss_single(logit1, covi_gt_1)
            # Average for this stage
            stage_loss = (loss_0 + loss_1) / 2.0
            stage_losses.append(stage_loss)

        if self.covi_multi_stage_avg:
            # Average across all stages
            return sum(stage_losses) / len(stage_losses)
        else:
            # Sum across all stages
            return sum(stage_losses)

    def logQ(self, gt_uv, pred_jts, sigma):
        assert self.q_distribution in ["laplace", "gaussian"]

        error = (pred_jts - gt_uv) / (sigma + 1e-9)

        if self.q_distribution == "laplace":
            loss_q = torch.log(sigma * 2) + torch.abs(error)
        else:
            loss_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

        return loss_q

    def compute_rle_loss(self, data, f_weight=1):
        gt_uv = data["target_uv"]
        gt_uv_weight = data["target_uv_weight"]

        if gt_uv_weight.sum() == 0:
            if (
                self.training
            ):  # this seldomly happen when training, since we pad prediction with gt
                logger.warning(
                    "assign a false supervision to avoid ddp deadlock")
                gt_uv_weight[0] = True
                f_weight = 0.0
            else:
                return None

        residual = True
        if residual:
            Q_logprob = self.logQ(
                gt_uv[gt_uv_weight], data["mask_coord"], data["mask_sigma"]
            )
            loss = Q_logprob + data["nf_loss"]

        return loss.mean() * f_weight

    @torch.no_grad()
    def compute_c_weight(self, data):
        """compute element-wise weights for computing coarse-level loss."""
        if "mask0" in data:
            c_weight = (
                data["mask0"].flatten(-2)[..., None]
                * data["mask1"].flatten(-2)[:, None]
            ).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data["conf_matrix"],
            data["conf_matrix_gt"],
            weight=c_weight,
        )
        loss = loss_c * self.loss_config["coarse_weight"]
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 1.5 DCAT multi-stage covisibility loss
        if self.dcat_enabled and "stage_matchability_logits" in data:
            stage_logits = data["stage_matchability_logits"]
            covi_gt_0 = data.get("covi_gt_0")
            covi_gt_1 = data.get("covi_gt_1")

            if stage_logits is not None and covi_gt_0 is not None and covi_gt_1 is not None:
                loss_covi = self.compute_covi_loss_multi_stage(
                    stage_logits, covi_gt_0, covi_gt_1
                )
                if loss_covi is not None:
                    loss += loss_covi * self.covi_weight
                    loss_scalars.update(
                        {"loss_covi": loss_covi.clone().detach().cpu()}
                    )

        # 1.6 Legacy single-stage covisibility loss (when DCAT disabled)
        elif "covi_logits_0" in data and "covi_gt_0" in data:
            loss_covi = (
                F.binary_cross_entropy_with_logits(
                    data["covi_logits_0"], data["covi_gt_0"], reduction='mean')
                + F.binary_cross_entropy_with_logits(
                    data["covi_logits_1"], data["covi_gt_1"], reduction='mean')
            ) / 2.0
            loss += loss_covi * self.covi_weight
            loss_scalars.update(
                {"loss_covi": loss_covi.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f = self.compute_rle_loss(
            data=data,
            f_weight=self.loss_config["fine_weight"],
        )
        if loss_f is not None:
            loss += loss_f
            loss_scalars.update(
                {"loss_f": min(loss_f.clone().detach().cpu(),
                               torch.tensor(1.0))}
            )
        else:
            assert self.training is False
            # 1 is the upper bound
            loss_scalars.update({"loss_f": torch.tensor(1.0)})

        loss_scalars.update({"loss": loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
