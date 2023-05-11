


</div>

## Prerequisite

Install the required python packages by:

```bash
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


