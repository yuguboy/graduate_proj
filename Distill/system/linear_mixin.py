from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_system import concat_all_ddp, aggregate_dict


class LinearMixin():
    # 这个方法用于前向传递（forward pass），即将输入数据x通过模型，得到模型的预测结果。
    def _forward_batch(self, x):
        return self.forward(x)

    # 用于线性模型的验证步骤。它接受一个批次数据batch和批次索引batch_idx，
    # 对输入数据进行前向传递，计算预测结果与标签之间的交叉熵损失（cross entropy loss），
    # 并将损失值记录到日志中。同时，该方法还返回一个字典，其中包含了预测结果（prob）和标签（gt）。
    def _linear_validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        mlp_preds, *_ = self._forward_batch(x)
        mlp_loss = torch.nn.functional.cross_entropy(mlp_preds, y)
        self.log("linear_loss", mlp_loss, on_epoch=True)
        # result.loss = mlp_loss
        prob = mlp_preds.softmax(dim=-1)
        gt = y
        return {"prob": prob, "gt": gt}

    # 用于线性模型的验证结束步骤。它接受一个包含多个批次的输出outputs，
    # 将每个批次的预测结果和标签汇总起来，计算整个验证集上的平均准确率（mean accuracy），
    # 并将结果记录到日志中。
    def _linear_validation_epoch_end(self, outputs, *args, **kwargs):
        outputs = aggregate_dict(outputs)
        epoch_probs = concat_all_ddp(outputs["prob"])
        epoch_gt = concat_all_ddp(outputs["gt"])
        epoch_preds = torch.argmax(epoch_probs, dim=-1)
        
        # 保存平均准确率
        mean_accuracy = accuracy(epoch_preds, epoch_gt)

        self.log("acc_mean", mean_accuracy)
