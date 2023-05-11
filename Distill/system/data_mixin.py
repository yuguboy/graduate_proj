from pytorch_lightning.core.lightning import LightningModule
from typing import MutableSequence


class LightningDataMixin(LightningModule):
    def __init__(self):
        super().__init__()

# 根据数据集的名字（字符串）或者数据集列表，计算所有数据集中的类别总数。
    def _cal_num_classes(self, dataset_name):
        def fn_num(dset):
            return self.dm.get_num_class(dset)

        if isinstance(dataset_name, str):
            return fn_num(dataset_name)
        elif isinstance(dataset_name, MutableSequence):
            return sum(fn_num(dset) for dset in dataset_name)
        else:
            return None

# 数据准备方法，这个方法由PyTorch Lightning调用一次，用来准备数据。它的实现可能包括数据下载、预处理、缓存等操作。
    def prepare_data(self):
        self.dm.prepare_data()

# 获取训练数据加载器，它调用一个数据管理器（DataModule）中的方法来获取数据加载器，同时传递一些参数，如训练器（pl_trainer）和是否使用分布式训练（use_ddp）等。
    def train_dataloader(self):
        return self.dm.train_dataloader(pl_trainer=self.trainer,
                                        use_ddp=self.trainer.use_ddp)

# 获取验证数据加载器，如果禁用了验证，返回None。
    def val_dataloader(self):
        if self.hparams.disable_validation:
            return None
        return self._eval_dataloader(self.hparams.data.val_dataset)

# 获取测试数据加载器，如果没有指定测试数据集，则返回验证数据加载器；否则返回一个指定的数据集的加载器。
    def test_dataloader(self):
        if self.hparams.data.test_dataset is None:
            return self.val_dataloader()
        else:
            return self._eval_dataloader(self.hparams.data.test_dataset)

# 根据评估模式（few_shot或linear）返回对应的数据加载器。
    def _eval_dataloader(self, dataset):
        if self.hparams.eval_mode == 'few_shot':
            return self._get_fewshot_loader(dataset)

        elif self.hparams.eval_mode == 'linear':
            return self._get_linear_loader(dataset)

# 获取适用于few-shot评估模式的数据加载器，传递一些参数，
# 如数据管理器（dm）、训练器（pl_trainer）、是否使用分布式训练（use_ddp）、
# 数据增强（aug）和数据集（dataset）等。
    def _get_fewshot_loader(self, dataset):
        return self.dm.get_fewshot_dataloader(pl_trainer=self.trainer,
                                              use_ddp=self.trainer.use_ddp,
                                              aug=self.hparams.data.val_aug,
                                              datasets=dataset)

# 获取适用于linear评估模式的数据加载器，
    def _get_linear_loader(self, dataset):
        base_loader = self.dm.get_simple_dataloader(
            dataset,
            aug=self.hparams.data.val_aug,
            pl_trainer=self.trainer,
            use_ddp=self.trainer.use_ddp,
            opt=self.hparams.data,
            shuffle=self.hparams.data.shuffle_val)
        return base_loader

# 批量大小属性，可以获取和设置批量大小（batch_size）。这个属性在其他方法中被使用，如获取数据加载器等。
    @property
    def batch_size(self) -> int:
        return self.hparams.data.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.hparams.data.batch_size = value