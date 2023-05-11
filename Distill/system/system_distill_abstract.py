import copy
from typing import MutableSequence
from torch import nn
from torch.functional import Tensor
from system.multiloader_mixin import MultiTrainLoaderMixin
from system import system_abstract
import torch
import abc
import torchmetrics
from torch.nn import functional as F
import math
from utils.IDM_module import IDM


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(MultiTrainLoaderMixin, system_abstract.LightningSystem):
    """ Abstract class """
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)
        if isinstance(self.hparams.data.dataset, MutableSequence) and len(
                self.hparams.data.dataset) == 1:
            self.hparams.data.dataset = self.hparams.data.dataset[0]
        self._create_model(self.num_classes)

        if self.hparams.ckpt_preload is not None:
            ckpt = torch.load(
                self.hparams.ckpt_preload,
                map_location=lambda storage, loc: storage)['state_dict']
            self.load_state_dict(ckpt, strict=False)

        self.create_system()
        self.load_IDM()
        
        
        # self.distill_loss = self.distill_loss(self.hparams.teacher_temp)
        self.ce_loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.divloss = DivLoss()
        self.bridge_prob_loss = BridgeProbLoss(num_classes = self.num_classes)
        self.bridge_feature_loss = BridgeFeatLoss()


    def create_system(self):

        self.sys_basic_block = self.feature.basic_layers
        self.sys_layers1 = self.feature.layers[0]
        self.sys_IDM = IDM(64)
        self.sys_student_layers2 = self.feature.layers[1]
        self.sys_student_layers3 = self.feature.layers[2]
        self.sys_student_layers4 = self.feature.layers[3]
        self.student_blocks = ConcatenatedBlocks(self.sys_student_layers2, 
                                                 self.sys_student_layers3, 
                                                 self.sys_student_layers4)
        
        self.teacher_basic_block = copy.deepcopy(self.sys_basic_block)
        self.teacher_layers1 = copy.deepcopy(self.sys_layers1)
        self.teacher_blocks = copy.deepcopy(self.student_blocks)

        self.student_header = (self.get_header())
        self.teacher_header = copy.deepcopy(self.student_header)


        self.sys_IDM.requires_grad_(True)

        self.sys_basic_block.requires_grad_(True)
        self.sys_layers1.requires_grad_(True)
        self.student_blocks.requires_grad_(True)
        self.student_header.requires_grad_(True)

        self.teacher_basic_block.requires_grad_(False)
        self.teacher_layers1.requires_grad_(False)
        self.teacher_blocks.requires_grad_(False)
        self.teacher_header.requires_grad_(False)
        
        
    def load_IDM(self, ckpt_path=None, prefix='sys_IDM'):
        ckpt_path = self.hparams.ckpt_preload
        print(ckpt_path)
        if ckpt_path is not None:
            ckpt = torch.load(
                ckpt_path,
                map_location=lambda storage, loc: storage)['state_dict']
            new_state = {}
            for k, v in ckpt.items():
                if f'{prefix}.' in k:
                    new_state[k.replace(f'{prefix}.', '')] = v
            self.sys_IDM.load_state_dict(new_state,
                                         strict=not self.hparams.load_flexible)
                                        # strict=False)

    # @abc.abstractmethod
    # def create_teacher(self):
    #     pass

    def forward(self, x):
        out = self.sys_basic_block(x)
        out = self.sys_layers1(out)
        out = self.student_blocks(out)
        out = self.student_header(out)
        return out
    
    def teacher_out(self, x):
        out = self.teacher_basic_block(x)
        out = self.teacher_layers1(out)
        out = self.teacher_blocks(out)
        out = self.teacher_header(out)
        return out
    
    def student_out(self, x):
        out = self.sys_basic_block(x)
        out = self.sys_layers1(out)
        out = self.student_blocks(out)
        out = self.student_header(out)
        return out

    def set_forward(self, *x_list):
        dims = [x.shape[0] for x in x_list]
        scores_all = self(torch.cat(x_list, dim=0))
        return scores_all.split(dims)
    

    # --------------------------------- training --------------------------------- #

    @abc.abstractmethod
    def _forward_loss(self, batch, batch_u):
        pass

    def training_step(self, batch, batch_idx=0):

        x, y = batch

        batch_train = (x, y)
        batch_u = self.get_unlabel_batch()

        loss_ce, loss_pseudo, loss_div, loss_bf, loss_pseudo_inter, top1, attention = self._forward_loss(batch_train, batch_u)

        l2 = 1
        l3 = 1
        if self.hparams.cosine_weight:
            l2 = self.get_cosine_weight()
            l3 = self.get_cosine_weight2()

        
        # loss = (0
        #         + 10  * self.hparams.lm_ce * loss_ce 
        #         + 10 * (l2) * self.hparams.lm_u * loss_pseudo
        #         + 0.1 * (0.1+0.9*l3) * loss_div
        #         + 0.1 * (0.1+0.9*l3) * loss_bf
        #         + 0.5 * (0.1+0.9*l3) * loss_pseudo_inter
        #         )
        
        loss = (0
                + 10  * self.hparams.lm_ce * loss_ce 
                + 10 * (l2) * self.hparams.lm_u * loss_pseudo
                + 0.1 * (l2) * loss_div
                + 0.1 * (l2) * loss_bf
                + 0.5 * (l2) * loss_pseudo_inter
                )

        
        mu = attention.mean(0)
        tqdm_dict = {
            "mo":self.hparams.mometum_update,
            "l3": l3,
            "l2": l2,
            "loss_train": loss,
            "l_ce": self.hparams.lm_ce * loss_ce,
            "l_u": self.hparams.lm_u * loss_pseudo,
            "l_div": loss_div,
            "l_bf": loss_bf,
            "loss_pseudo_inter": loss_pseudo_inter,
            "top1": top1,
            "attention_on_source":mu[0]
        }
        print(tqdm_dict)

        self.log_dict(tqdm_dict, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # --------------------------------- utilities -------------------------------- #\
    def cross_entropy(self, logits, y_gt) -> Tensor:
        return cross_entropy(logits, y_gt)

    def get_header(self) -> nn.Module:
        header = self.classifier
        return header

    
    def distill_loss(self, student_out, teacher_out):
        teacher_out /= self.hparams.teacher_temp
        loss = self.cross_entropy(student_out, teacher_out.softmax(dim=-1))
        return loss

    
    def get_cosine_weight(self):
        total_steps = self.trainer.max_epochs * self.trainer.num_training_batches
        current_step = self.trainer.global_step
        multiplier = min(
            1, 0.75 * (1 - math.cos(math.pi * current_step / total_steps)))
        return multiplier
    
    def get_cosine_weight2(self):
        total_steps = self.trainer.max_epochs * self.trainer.num_training_batches
        current_step = self.trainer.global_step
        if current_step*12 < total_steps*0.6:
            multiplier = min(
                1, 0.75 * (1 - math.cos(math.pi * current_step*12 / total_steps)))
        else:
            multiplier = 1
        return multiplier

def cross_entropy(logits, y_gt) -> Tensor:
    if len(y_gt.shape) < len(logits.shape):
        return F.cross_entropy(logits, y_gt, reduction='mean')
    else:
        return (-y_gt * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


class DivLoss(nn.Module):
    def __init__(self, ):
        super(DivLoss, self).__init__()
        
    def forward(self, scores):
        mu = scores.mean(0)
        # print(mu)
        std = ((scores-mu)**2).mean(0,keepdim=True).clamp(min=1e-12).sqrt()
        loss_std = -std.sum()
        # print(loss_std)

        return loss_std
class BridgeFeatLoss(nn.Module):
    def __init__(self):
        super(BridgeFeatLoss, self).__init__()

    def forward(self, feats_s, feats_t, feats_mixed, lam):

        dist_mixed2s = ((feats_mixed-feats_s)**2).sum(1, keepdim=True)
        dist_mixed2t = ((feats_mixed-feats_t)**2).sum(1, keepdim=True)
        
        dist_mixed2s = dist_mixed2s.clamp(min=1e-12).sqrt()
        dist_mixed2t = dist_mixed2t.clamp(min=1e-12).sqrt()
        # print(dist_mixed2s,dist_mixed2t)

        dist_mixed = torch.cat((dist_mixed2t, dist_mixed2s), 1)
        lam_dist_mixed = (lam*dist_mixed).sum(1, keepdim=True)
        loss = lam_dist_mixed.mean()
        # print(loss)
        # assert(1==0)
        return loss



class BridgeProbLoss(nn.Module):

    def __init__(self, num_classes, epsilon=0.1):
        super(BridgeProbLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.device_num = torch.cuda.device_count()

    def forward(self, inputs, targets, lam):

        inputs = inputs.view(1, -1, inputs.size(-1))
        inputs_s, inputs_t, inputs_mixed = inputs.split(inputs.size(0) // 3, dim=0)
        inputs_ori = torch.cat((inputs_s, inputs_t), 1).view(-1, inputs.size(-1))
        inputs_mixed = inputs_mixed.contiguous().view(-1, inputs.size(-1))
        log_probs_ori = self.logsoftmax(inputs_ori)
        log_probs_mixed = self.logsoftmax(inputs_mixed)

        targets = torch.zeros_like(log_probs_ori).scatter_(1, targets.unsqueeze(1), 1)
        # print(targets.shape)
        targets = targets.view(1, -1, targets.size(-1))
        targets_s, targets_t = targets.split(targets.size(1) // 2, dim=0)
        targets_s = targets_s.contiguous()
        targets_t = targets_t.contiguous()

        targets_s = targets_s.contiguous().view(-1, targets.size(-1))
        targets_t = targets_t.contiguous().view(-1, targets.size(-1))

        targets = targets.view(-1, targets.size(-1))
        soft_targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        lam = lam.view(-1, 1)
        soft_targets_mixed = lam*targets_s+(1.-lam)*targets_t
        soft_targets_mixed = (1 - self.epsilon) * soft_targets_mixed + self.epsilon / self.num_classes
        loss_ori = (- soft_targets*log_probs_ori).mean(0).sum()
        loss_bridge_prob = (- soft_targets_mixed*log_probs_mixed).mean(0).sum()

        return loss_ori, loss_bridge_prob
    
    
class ConcatenatedBlocks(nn.Module):
    def __init__(self, block2, block3, block4, flatten=True):
        super(ConcatenatedBlocks, self).__init__()
        self.block2 = block2
        self.block3 = block3
        self.block4 = block4

    def forward(self, x):
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x
