# Inter-Target Distill Network for CDFSL

## Abstract

​	In practical image classification problems, one often lacks sufficient prior knowledge in certain domains and acquiring large amounts of labeled data to implement supervised learning is very difficult. One route to improve the classification performance of the model on these domains is to acquire sufficient prior knowledge on a sample of categories with sufficient data and then use a small number of samples on the target category to quickly learn good classification criteria. However, recent research have found that simply using few-shot learning methods can't achieve good performance when there is a significant domain gap between the source and target domain datasets, which leads to a new research direction - cross-domain few-shot learning. Unlike the classical few-shot learning, the cross-domain few-shot learning problem assumes the existence of domain gap between the source and target datasets, which is more in line with the practical application requirements of few-shot learning. The key to solve the cross-domain few-shot learning problem is to utilize the labeled data in the source domain to improve the generalization performance of the model in the target domain. It is noted that the existing methods often have difficulty in obtaining high generalization accuracy when the domain gap is large. In order to improve the generalization performance of the model under large domain gap, this paper proposes a method of dynamic distillation in the intermediate and target domains. Specifically, the method implements supervised training using source domain data, generates intermediate domain samples in the feature space using the intermediate domain module, and simultaneously implements self-supervised learning using a dynamic teacher network to generate pseudo-labels for the intermediate and target domain samples. The test results on the training model show that the proposed method can achieve performance beyond other methods when the domain gap is large, and the results are also competitive on data sets with small domain gap.

**Keywords: Cross-Domain Few-shot learning; Self-supervise learning; Domain Adaptation; Knowledge Distillation**		

## Method $ Model

![model](model.png)



## Prerequisite

Install the required python packages by:

```bash
cd Distill
pip install -r requirements.txt
```

## Dataset

Download the datasets from [here](https://drive.google.com/drive/folders/1X7WacUWKjZpjR2qo0gvaPY6dxkqtjGtz?usp=sharing), and keep them in the `data/` directory.

## Pretraining

Download the cross-entropy pretrained model (download from [here](https://drive.google.com/drive/folders/1T6QzEnAnbw4-FljldU03YJ84RZUfWhjm?usp=sharing)) on mini-ImageNet in `ckpt/ce_miniImageNet_resnet10`.

Or, train by running:

```bash
python main.py system=ce  backbone=resnet10 data.dataset=miniImageNet_train  model_name=ce_miniImageNet_resnet10 trainer.gpus=4
```

## Training

To train on mini-ImageNet and unlabeled target images, run the following command:

```bash
python main.py system=ce_distill_ema_sgd trainer.gpus=1 backbone=resnet10 \
  data.val_dataset=EuroSAT_test data.test_dataset=null print_val=false \
   trainer.log_every_n_steps=-1  unlabel_params.dataset=EuroSAT_train \
    data.num_episodes=600  trainer.progress_bar_refresh_rate=0 print_val=false \
     launcher.gpus=1 pretrained=true model_name=try_EuroSAT
```

Change `EuroSAT` to other dataset to train on `CropDisese`, `ChestX`, `ISIC`.

Model will be saved in `ckpt/dynamic_cdfsl_EuroSAT/last.ckpt`

## Few-shot evaluation

For 5-way 5-shot evaluation on EuroSAT dataset:

```bash
python main.py system=few_shot  data.test_dataset=EuroSAT_test  ckpt=[pretrained-checkpoint]
```

## Acknowledgement

- [cdfsl-benchmark](https://github.com/IBM/cdfsl-benchmark)
- [Dynamic Distillation Network for Cross-Domain Few-Shot Recognition with Unlabeled Data](https://git.io/Jilgs)
- [IDM: An Intermediate Domain Module for Domain Adaptive Person Re-ID](https://github.com/SikaStar/IDM)
