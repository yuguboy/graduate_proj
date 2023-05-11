import torch
from torchmetrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from system import system_distill_abstract
import copy
from typing import Any
import torch.distributed as dist
from utils.IDM_module import IDM
from torch.functional import Tensor
import numpy as np

import torch
from torchvision import transforms
 
toPIL = transforms.ToPILImage() 




torch.set_printoptions(threshold=np.inf)


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(system_distill_abstract.LightningSystem):
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)



        if self.hparams.apply_center:
            self.register_buffer("center", torch.zeros(1, self.num_classes))


    # --------------------------------- training --------------------------------- #
# # same as paer:
#     def _forward_loss(self, batch, batch_u):

#         x, y = batch
#         (x_u_w, x_u_s), *_ = batch_u

#         inputs = torch.cat((x, x_u_s),dim=0)

#         inputs = self.sys_basic_block(inputs)
#         inputs = self.sys_layers1(inputs)

#         out_idm , attention_lam = self.sys_IDM(inputs)

#         # inter pseudo loss
#         bs = out_idm.size(0)
#         assert (bs%3==0)
#         split = torch.split(out_idm, int(bs/3), 0)
#         G_s, G_t, G_inter = split[0], split[1], split[2]
#         f_inter_pseudo =self.teacher_blocks(G_inter)
#         g_inter_pseudo = self.teacher_header(f_inter_pseudo)

#         inputs = torch.cat((G_s,G_t), dim=0)

#         inputs = self.student_blocks(inputs)
#         bs = inputs.size(0)
#         assert (bs%2==0)
#         split = torch.split(inputs, int(bs/2), 0)
#         f_s, f_t = split[0], split[1]

#         inputs = self.student_header(inputs)
#         bs = inputs.size(0)
#         assert (bs%2==0)
#         split = torch.split(inputs, int(bs/2), 0)
#         g_s, g_t = split[0], split[1]

#         # ce loss
#         loss_ce = self.ce_loss(g_s, y)
#         top1 = self.train_acc(g_s.argmax(dim=-1), y)


#         # pseudo loss
#         if self.hparams.unlabel_params.no_stop_grad is False:
#             torch.set_grad_enabled(False)

#         logit_pseudo = (self.teacher_out(x_u_w))
#         # print(logit_pseudo)
#         logit_pseudo = logit_pseudo.detach()


#         if self.hparams.unlabel_params.no_stop_grad is False:
#             torch.set_grad_enabled(True)

#         print(attention_lam)

#         # IDM Loss
#         loss_pseudo = self.distill_loss(g_t, logit_pseudo)
#         loss_pseudo_inter = 0
#         loss_div = 0
#         # _, loss_bprob = self.bridge_prob_loss(prob, targets, attention_lam[:,0].detach())
#         loss_bf = 0
#         # print(loss_ce, loss_pseudo, loss_div, loss_bf, loss_bprob, top1)

#         return  loss_ce, loss_pseudo, loss_div, loss_bf, loss_pseudo_inter, top1, attention_lam

# # before 4_9
#     def _forward_loss(self, batch, batch_u):

#         x, y = batch
#         (x_u_w, x_u_s), *_ = batch_u 

#         inputs = torch.cat((x, x_u_s),dim=0)

#         inputs = self.sys_basic_block(inputs)
#         inputs = self.sys_layers1(inputs)

#         out_idm , attention_lam = self.sys_IDM(inputs)

#         inputs = self.student_blocks(out_idm)
#         bs = inputs.size(0)
#         assert (bs%3==0)
#         split = torch.split(inputs, int(bs/3), 0)
#         f_s, f_t, f_inter = split[0], split[1], split[2]

#         inputs = self.student_header(inputs)
#         bs = inputs.size(0)
#         assert (bs%3==0)
#         split = torch.split(inputs, int(bs/3), 0)
#         g_s, g_t, g_inter = split[0], split[1], split[2]

#         prob = torch.cat((g_s, g_t, g_inter),dim=0)

#         # ce loss
#         loss_ce = self.ce_loss(g_s, y)
#         top1 = self.train_acc(g_s.argmax(dim=-1), y)

#         # pseudo loss
#         if self.hparams.unlabel_params.no_stop_grad is False:
#             torch.set_grad_enabled(False)

#         logit_pseudo = (self.teacher_out(x_u_w))
#         # if self.hparams.apply_center:
#         #     logit_pseudo = logit_pseudo - self.center
#         #     self.update_center(logit_pseudo.clone().detach())
#         # logit_pseudo = logit_pseudo.detach()

#         if self.hparams.unlabel_params.no_stop_grad is False:
#             torch.set_grad_enabled(True)

#         _, y_t = torch.max(logit_pseudo, 1)

#         targets = torch.cat((y, y_t), dim=0)

#         # print(attention_lam)

#         # IDM Loss
#         loss_pseudo = self.distill_loss(g_t, logit_pseudo)
#         loss_div = self.divloss(attention_lam)
#         _, loss_bprob = self.bridge_prob_loss(prob, targets, attention_lam[:,0].detach())
#         loss_bf = self.bridge_feature_loss(f_s, f_t, f_inter, attention_lam)
#         # print(loss_ce, loss_pseudo, loss_div, loss_bf, loss_bprob, top1)

#         return  loss_ce, loss_pseudo, loss_div, loss_bf, loss_bprob, top1, attention_lam

# # improve 4_14 : using teacher net to get inter-domain labels
#     def _forward_loss(self, batch, batch_u):

#         x, y = batch
#         (x_u_w, x_u_s), *_ = batch_u

#         # get G_inter for week-aug version data
#         inputs = torch.cat((x, x_u_s), dim=0)

#         inputs = self.sys_basic_block(inputs)
#         inputs = self.sys_layers1(inputs)

#         out_idm , attention_lam = self.sys_IDM(inputs)

#         bs = out_idm.size(0)
#         assert (bs%3==0)
#         split = torch.split(out_idm, int(bs/3), 0)
#         G_s, G_t, G_inter = split[0], split[1], split[2]
    
#         # --------------------------------------------------- 
#         inputs = torch.cat((G_s,G_t,G_inter), dim=0)
#         inputs = self.student_blocks(inputs)
#         bs = inputs.size(0)
#         assert (bs%3==0)
#         split = torch.split(inputs, int(bs/3), 0)
#         f_s, f_t, f_inter = split[0], split[1], split[2]

#         inputs = self.student_header(inputs)
#         bs = inputs.size(0)
#         assert (bs%3==0)
#         split = torch.split(inputs, int(bs/3), 0)
#         g_s, g_t, g_inter = split[0], split[1], split[2]

#         # ce loss
#         loss_ce = self.ce_loss(g_s, y)
#         top1 = self.train_acc(g_s.argmax(dim=-1), y)

#         # pseudo loss
#         if self.hparams.unlabel_params.no_stop_grad is False:
#             torch.set_grad_enabled(False)

#         logit_pseudo = (self.teacher_out(x_u_w))

#         logit_pseudo = logit_pseudo.detach()

#         # inter pseudo loss
#         f_inter_pseudo =self.teacher_blocks(G_inter)
#         g_inter_pseudo = self.teacher_header(f_inter_pseudo)

#         g_inter_pseudo = g_inter_pseudo.detach()


#         if self.hparams.unlabel_params.no_stop_grad is False:
#             torch.set_grad_enabled(True)   

#         # print(attention_lam)
#         # g_inter_pseudo = g_s + g_t*1.1

#         # IDM Loss
#         loss_pseudo = self.distill_loss(g_t, logit_pseudo)
#         loss_pseudo_inter = self.distill_loss(g_inter, (g_inter_pseudo)/20)
#         loss_pseudo_inter=(loss_pseudo_inter.requires_grad_())
#         loss_div = self.divloss(attention_lam)
#         # _, loss_bprob = self.bridge_prob_loss(prob, targets, attention_lam[:,0].detach())
#         loss_bf = self.bridge_feature_loss(f_s, f_t, f_inter, attention_lam)
        

#         return  loss_ce, loss_pseudo, loss_div, loss_bf, loss_pseudo_inter, top1, attention_lam
   
   # 4_18: duel out_idm
   
   # --------------------------step 1------------
    def _forward_loss(self, batch, batch_u):

        x, y = batch
        (x_u_w, x_u_s), *_ = batch_u


        # get G_inter for week-aug version data
        inputs = torch.cat((x, x_u_w), dim=0)

        inputs = self.sys_basic_block(inputs)
        inputs = self.sys_layers1(inputs)

        out_idm , attention_lam = self.sys_IDM(inputs)

        bs = out_idm.size(0)
        assert (bs%3==0)
        split = torch.split(out_idm, int(bs/3), 0)
        __, ___, G_inter_w = split[0], split[1], split[2]
        
        # get G_inter_s for strong-aug version data
        inputs = torch.cat((x, x_u_s), dim=0)

        inputs = self.sys_basic_block(inputs)
        inputs = self.sys_layers1(inputs)

        out_idm , attention_lam = self.sys_IDM(inputs)

        bs = out_idm.size(0)
        assert (bs%3==0)
        split = torch.split(out_idm, int(bs/3), 0)
        G_s, G_t, G_inter_s = split[0], split[1], split[2]

        # --------------------------------------------------- 

        
        inputs = torch.cat((G_s,G_t), dim=0)
        inputs = self.student_blocks(inputs)
        bs = inputs.size(0)
        assert (bs%2==0)
        split = torch.split(inputs, int(bs/2), 0)
        f_s, f_t = split[0], split[1]

        inputs = self.student_header(inputs)
        bs = inputs.size(0)
        assert (bs%2==0)
        split = torch.split(inputs, int(bs/2), 0)
        g_s, g_t = split[0], split[1]

        # ce loss
        loss_ce = self.ce_loss(g_s, y)
        top1 = self.train_acc(g_s.argmax(dim=-1), y)

        # pseudo loss
        # if self.hparams.unlabel_params.no_stop_grad is False:
        #     torch.set_grad_enabled(False)

        logit_pseudo = (self.teacher_out(x_u_w))
        logit_pseudo = logit_pseudo.detach()
        
        if self.hparams.apply_center:
            logit_pseudo = logit_pseudo - self.center
            self.update_center(logit_pseudo.clone().detach())

        # inter pseudo loss
        f_inter = self.student_blocks(G_inter_s)
        g_inter = self.student_header(f_inter)
        
        f_inter_pseudo =self.teacher_blocks(G_inter_w)
        g_inter_pseudo = self.teacher_header(f_inter_pseudo)


        # print(out_idm.requires_grad)
        # print(G_inter_s.requires_grad)
        # print(f_inter.requires_grad)
        # print(g_inter.requires_grad)
        if self.hparams.unlabel_params.no_stop_grad is False:
            torch.set_grad_enabled(True)

        # print(attention_lam)

        # IDM Loss
        loss_pseudo = self.distill_loss(g_t, logit_pseudo)
        loss_pseudo_inter = self.distill_loss(g_inter, (g_inter_pseudo)/5)
        loss_div = self.divloss(attention_lam)
        # _, loss_bprob = self.bridge_prob_loss(prob, targets, attention_lam[:,0].detach())
        loss_bf = self.bridge_feature_loss(f_s, f_t, f_inter, attention_lam)

        return  loss_ce, loss_pseudo, loss_div, loss_bf, loss_pseudo_inter, top1, attention_lam

    # # -----------------step 2:unsupervised------------
    # def _forward_loss(self, batch, batch_u):

    #     # x, y = batch
    #     (x_u_w, x_u_s), *_ = batch_u
        
    #     logit_u_s = self.student_out(x_u_s)
    #     logit_pseudo = self.teacher_out(x_u_w)
    #     logit_pseudo = logit_pseudo.detach()

    #     # ce loss
    #     loss_ce = 0
    #     top1 = 0

    #     # pseudo loss



        
    #     # if self.hparams.unlabel_params.no_stop_grad is False:
    #     #     torch.set_grad_enabled(False)
            
    #     if self.hparams.apply_center:
    #         logit_pseudo = logit_pseudo - self.center
    #         self.update_center(logit_pseudo.clone().detach())

    #     # IDM Loss
    #     loss_pseudo = self.distill_loss(logit_u_s, logit_pseudo)
    #     loss_pseudo.requires_grad_(True)
    #     loss_pseudo_inter = 0
    #     loss_div = 0
    #     # _, loss_bprob = self.bridge_prob_loss(prob, targets, attention_lam[:,0].detach())
    #     loss_bf = 0
    #     attention_lam = 0
    #     return  loss_ce, loss_pseudo, loss_div, loss_bf, loss_pseudo_inter, top1, attention_lam

    def on_train_batch_start(self, batch: Any, batch_idx: int,
                             dataloader_idx: int) -> None:
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)
        
        # EMA update for the teacher
        with torch.no_grad():
            m = self.hparams.mometum_update  # momentum parameter
            for param_q, param_k in zip(self.student_blocks.parameters(),
                                        self.teacher_blocks.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        
        with torch.no_grad():
            m = self.hparams.mometum_update  # momentum parameter
            for param_q, param_k in zip(self.student_header.parameters(),
                                        self.teacher_header.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        with torch.no_grad():
            m = self.hparams.mometum_update  # momentum parameter
            for param_q, param_k in zip(self.sys_basic_block.parameters(),
                                        self.teacher_basic_block.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        with torch.no_grad():
            m = self.hparams.mometum_update  # momentum parameter
            for param_q, param_k in zip(self.sys_layers1.parameters(),
                                        self.teacher_layers1.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)        
        
        self.sys_IDM.requires_grad_(True)

        self.sys_basic_block.requires_grad_(True)
        self.sys_layers1.requires_grad_(True)
        self.student_blocks.requires_grad_(True)
        self.student_header.requires_grad_(True)

        self.teacher_basic_block.requires_grad_(False)
        self.teacher_layers1.requires_grad_(False)
        self.teacher_blocks.requires_grad_(False)
        self.teacher_header.requires_grad_(False)


    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        # self.teacher.requires_grad_(False)
        self.teacher_blocks.requires_grad_(False)
        self.teacher_header.requires_grad_(False)

    

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))
        # ema update
        self.center = self.center * self.hparams.center_momentum + batch_center * (
            1 - self.hparams.center_momentum)

    def get_feature_extractor(self):
        
        return self.feature
        
# class DivLoss(nn.Module):
#     def __init__(self, ):
#         super(DivLoss, self).__init__()
        
#     def forward(self, scores):
#         mu = scores.mean(0)
#         # std = ((scores-mu)**2).mean(0,keepdim=True).clamp(min=1e-12).sqrt()
#         std = ((scores-mu)**2).mean(0,keepdim=True)
#         loss_std = -std.sum()
#         return loss_std

def cross_entropy(logits, y_gt) -> Tensor:
    if len(y_gt.shape) < len(logits.shape):
        return F.cross_entropy(logits, y_gt, reduction='mean')
    else:
        return (-y_gt * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()