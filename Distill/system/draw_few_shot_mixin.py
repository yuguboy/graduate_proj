


from utils import utils_system
import torch
import torch.nn as nn
import numpy as np
import torch
import numpy as np
from torchmetrics.functional import accuracy
from pytorch_lightning.utilities.distributed import rank_zero_only
from system import utils_finetune as ftune
from typing import MutableSequence


class FewShotMixin():
    # 用于初始化一个 AverageMeter 列表，这个列表会记录测试集上的平均准确率。
    def _init_meter(self) -> None:
        if self.testing:
            datasets = self.hparams.data.test_dataset
        else:
            datasets = self.hparams.data.val_dataset

        self.val_names = datasets if isinstance(
            datasets, MutableSequence) else [datasets]

        self.val_len = 1 if not isinstance(datasets,
                                           MutableSequence) else len(datasets)

        self.acc_meter = [
            utils_system.AverageMeter(use_ddp=self.trainer.use_ddp)
            for _ in range(self.val_len)
        ]

    # 用于进行 Few-Shot 验证。
    # 它首先调用 few_shot_finetune 方法来获取 Few-Shot 的准确率，然后根据需要打印准确率和记录准确率。
    def few_shot_validation_step(self, batch, batch_idx, dataset_idx=0):
        if not hasattr(self, 'acc_meter'):
            self._init_meter()

        out = self.few_shot_finetune(batch, dataset_idx)
        if self.hparams.print_val:
            self.loguru_log(
                f"val :: ({self.val_names[dataset_idx]}) : ({batch_idx}) : {out['acc']:.4f}, , acc: {out['acc']:.4f}, acc_mean: {out['acc_mean']:.4f}",
                level="DEBUG")

        if hasattr(self.logger, "log_csv"):
            self.logger.log_csv(
                {
                    "dataset": self.val_names[dataset_idx],
                    "accuracy": out['acc'],
                    "mean": out['acc_mean']
                },
                step=batch_idx)

        return out

    # 在验证结束时计算平均准确率和标准差。
    # 这个方法会重置 AverageMeter 列表，并返回一个包含平均准确率的字典。
    def few_shot_val_end(self, outputs):
        mean_val = []
        mean_std = []
        # max_val = []
        for dataset_idx in range(self.val_len):
            acc_mean = self.acc_meter[dataset_idx].mean
            acc_std = self.acc_meter[dataset_idx].std

            if self.logger is not None:
                self.loguru_log(
                    f"Test Acc ({self.val_names[dataset_idx]}) = {acc_mean:4.4f} +- {acc_std:4.4f}",
                    level="DEBUG")
            self.acc_meter[dataset_idx].reset()
            mean_val.append(acc_mean)
            mean_std.append(acc_std)

            if dataset_idx == 0:
                tqdm_dict = {"acc_mean": acc_mean, "acc_std": acc_std}
                self.log_dict(tqdm_dict, prog_bar=False)

        acc_mean = torch.mean(torch.stack(mean_val))
        acc_std = torch.mean(torch.stack(mean_std))
        tqdm_dict = {"acc_mean_full": acc_mean}
        self.log_dict(tqdm_dict, prog_bar=True)
        return tqdm_dict

    # 用于打印日志，如果使用了 LightningLoggerBase，
    # 则使用它的 log 方法，否则使用 experiment.log 方法
    @rank_zero_only
    def loguru_log(self, msg, level="INFO"):
        try:
            self.logger.log(msg)
        except AttributeError:
            self.logger.experiment.log(msg)

    #  从batch 中获取 Few-Shot 所需的数据
    def get_fewshot_batch(self, batch):
        x, y = batch
        self.n_way = self.hparams.n_way
        self.n_shot = self.hparams.n_shot

        self.n_query = x.size(1) - self.n_shot
        labels_dict_map = dict(
            zip(y[:, 0].data.cpu().numpy(), range(self.n_way)))

        # support set
        y_support = torch.from_numpy(np.repeat(range(
            self.n_way), self.n_shot)).to(self.device,
                                          non_blocking=True)  # (25,)

        x_support = x[:, :self.n_shot, :, :, :].contiguous().view(
            self.n_way * self.n_shot,
            *x.size()[2:])  # (25, 3, 224, 224)

        # query set
        x_query = x[:, self.n_shot:, :, :, :].contiguous().view(
            self.n_way * self.n_query,
            *x.size()[2:])

        y_query = torch.arange(self.n_way, device=self.device,
                               dtype=x.dtype).repeat_interleave(self.n_query)

        return x_support, y_support, x_query, y_query, labels_dict_map, y

    # 进行 Few-Shot 训练，返回一个包含准确率和平均准确率的字典
    def few_shot_finetune(self, batch, dataset_idx=0):
        x_support, y_support, x_query, y_query, *_ = self.get_fewshot_batch(
            batch)

        encoder = self.get_feature_extractor()

        topk_ind, _ = ftune.LR(encoder,
                               x_support,
                               y_support,
                               x_query,
                               y_query,
                               norm=self.hparams.eval.fine_tune.use_norm)

        _val = accuracy(topk_ind, y_query.long())
        self.acc_meter[dataset_idx].add(_val, n=1)

        return {
            "acc": self.acc_meter[dataset_idx].value,
            "acc_mean": self.acc_meter[dataset_idx].mean
        }


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim, n_way)
        # self.fc = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(),
        #                         nn.Linear(dim // 2, n_way))

    def forward(self, x):
        x = self.fc(x)
        return x
