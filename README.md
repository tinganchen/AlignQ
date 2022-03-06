# AlignQ: Alignment Quantization with ADMM-based Correlation Preservation
CVPR 2022 Accepted Paper - Quantization, Efficient Inference, Data Alignment

<img src="img/motivation.png" width="500" height="250">

## Requirements

* python3
* pytorch==1.7.1
* cudatoolkit==11.0.221 
* numpy==1.19.2
* tensorboardx==1.4

## Implementation

### CDF alignment quantization

* **Motivation**: our main contribution is to diminish the quantization error from the mismatch of training/testing data (non-i.i.d)

* **Method**: align data to the same uniform space by CDF transformation

* e.g. 8-bit ResNet-20 on CIFAR-10.

```shell
cd cdf_alignment/resnet-20-cifar-10/
```
Pretrain and save models to the path cdf_alignment/resnet-20-cifar-10/pretrained/.

```shell
python3 main.py --job_dir "experiment/ours/resnet/t_8bit_pre32bit" --method "ours" --source_dir "pretrained" --source_file "res20_32bit/model_best.pt" --arch resnet --bitW 8 --abitW 8 --target_model "resnet20_quant" --source_model "resnet20" --num_epochs 200 --train_batch_size 128 --eval_batch_size 100 --lr 0.04 --lr_gamma 0.1 --lr_decay_steps [80, 120] --momentum 0.9 --weight_decay 0.0001 --lam 1 --lam2 4 --act_range 2 --print_freq 200
```


### CDF alignment quantization with ADMM-based correlation preservation

* **Motivation**: to further improve the performance, we regularize the data correlations during the quantization process

* **Method**: minimize the changes in data correlations by ADMM optimization

* e.g. 8-bit DANN on Office-31 (D->W).

```shell
cd cdf_alignment_admm/dann_office/
```
Pretrain and save models to the path cdf_alignment_admm/dann_office/pretrained/.

```shell
python3 main.py --src_data "dslr" --tgt_data "webcam" --train_split "True" --job_dir "experiment/ours/resnet/t_8bit_pre32bit" --method "ours" --source_dir "pretrained" --source_file "dslr_webcam/dw_32bit/model_best.pt" --arch resnet --bitW 8 --abitW 8 --model "resnet50_dann" --num_epochs 200 --train_batch_size 28 --eval_batch_size 28 --lr 0.001 --lr_gamma 0.1 --lr_decay_steps [80, 120] --momentum 0.9 --weight_decay 0.0005 --lam 1 --lam2 4 --act_range 2 --print_freq 10
```

## Citation
