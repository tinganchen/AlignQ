# AlignQ: Domain Alignment Quantization with ADMM-based Contrastive Relation Preservation
Quantization, Domain Adaptation (Transfer Learning), Efficient Inference, Neural Network ([PDF](https://docs.google.com/presentation/d/19Pqz1WJYfiBvaYcWJWO8zRks63ZnlXU6NTADbT2wCZE/edit?usp=sharing))

<img src="img/contributions.jpg" width="700" height="250">

## Requirements

* python3
* pytorch==1.7.1
* cudatoolkit==11.0.221 
* numpy==1.19.2
* tensorboardx==1.4

## Pretrained weights download

* Single-domain (CIFAR-10)
  - [ResNet-20](https://drive.google.com/drive/folders/1WwfrNrsDHZl-kKONcXHUUnw3mNpCzh6k?usp=sharing)
  - [ResNet-56](https://drive.google.com/drive/folders/1OS5gVNOq0ZGdOr17Q2thM7oCpdrK0-Hy?usp=sharing)
  - [DenseNet-40](https://drive.google.com/drive/folders/1PPaUzZZyEa4mvV_4k-eHqo87ZBYb28TO?usp=sharing)

* Cross-domain (Digits)
  - [DANN (VGG-2)](https://drive.google.com/drive/folders/1g_XwptZf5_-En0aR9NJTrYfJ_lRbLFB2?usp=sharing)

* Cross-domain (Office-31)
  - [DANN (ResNet-50)](https://drive.google.com/drive/folders/1IFqIp_Kg-RpXsN8B2oUYH-QO-9jqczI3?usp=sharing)
  - [DSAN (ResNet-50)](https://drive.google.com/drive/folders/1bWQr0LZJVJmfvdLIjD1yU16yBhL-nZ_f?usp=sharing)


## Experiment

### Single-domain tasks

* e.g. 8-bit ResNet-20 on CIFAR-10.

```shell
cd AlignQ/single_domain/resnet-20-cifar-10/cdf_alignment/
```

```shell
python3 main.py --job_dir "experiment/ours/resnet/t_8bit_pre32bit" --method "ours" --source_dir "pretrained" --source_file "res20_32bit/model_best.pt" --arch resnet --bitW 8 --abitW 8 --target_model "resnet20_quant" --source_model "resnet20" --num_epochs 300 --train_batch_size 128 --eval_batch_size 100 --lr 0.04 --lr_gamma 0.1 --lr_decay_steps [80, 120] --momentum 0.9 --weight_decay 0.0001 --lam 1 --lam2 4 --act_range 2 --print_freq 200
```
Download the pretrained weights resnet-20-cifar-10/pretrained/ to the path AlignQ/single_domain/resnet-20-cifar-10/cdf_alignment/pretrained/.

### Cross-domain tasks

* e.g. 8-bit DANN (DAQ + CRP) on Office-31 (D->W).

```shell
cd AlignQ/cross_domain/dann_office/daq_crp/
```

```shell
python3 main.py --src_data "dslr" --tgt_data "webcam" --train_split "True" --job_dir "experiment/ours/resnet/t_8bit_pre32bit" --method "ours" --source_dir "pretrained" --source_file "dslr_webcam/dw_32bit/model_best.pt" --arch resnet --bitW 8 --abitW 8 --model "resnet50_dann" --num_epochs 100 --train_batch_size 28 --eval_batch_size 28 --lr 0.001 --lr_gamma 0.1 --lr_decay_steps [80, 120] --momentum 0.9 --weight_decay 0.0005 --lam 1 --lam2 4 --act_range 2 --print_freq 10
```
Download the pretrained weights dann_office/pretrained/ to the path AlignQ/cross_domain/dann_office/daq_crp/pretrained/.

### Compared with the baseline and the state-of-the-art quantization methods

* --method "uniform"
* --method "dorefa"
* --method "bwn"
* --method "bwnf"
* --method "lsq"
* --method "llsq"
* --method "apot"


## Detailed implementation. 
* The weights and the activations are simultaneously quantized to the same bits.

* For the single-domain tasks & the simple cross-domain task - DANN on Digits, the effectiveness of 
  - DAQ (CDF alignment) process & 
  - gradient approximation <br> are verified.
  
  - The codes are under the directory <cdf_alignment/>.
  
* For the other cross-domain tasks, the effectiveness of 
  - DAQ (CDF alignment) process & 
  - gradient approximation &
  - CRP (correlation preservation under ADMM optimization) <br> are verified.
  
  - The codes are under the directories, <cdf_alignment/> (only DAQ) and <daq_crp/> (DAQ + CRP).
  
* The baseline and the state-of-the-art quantization methods are also implemented, where the codes are referred to the public github repositories.

### Single-domain tasks

#### CIFAR-10.

Model                | Bit-width   | Pretrained Bit-width | DAQ    | Gradient Approx.  | CRP  | Hyperparameters 
---                  |:---:        |:---:                 |:---:   |:---:              |:---: |---                   
ResNet-20            | 8   |32    |V       |V                  |      | --target_model "resnet20_quant" --source_model "resnet20" --num_epochs 300 <br> --train_batch_size 128 <br> --test_batch_size 100 <br> --lr 0.1  <br> --lr_decay_steps [80, 120] <br> --lr_gamma 0.1 <br> --momentum 0.9  <br> --weight_decay 0.0001  <br> --lam 1  <br> --lam2 4  <br> --act_range 2  <br> --print_freq 200
ResNet-20            | 4 <br><br> 3 <br><br> 2   | 8 <br><br> 4 <br><br> 3  |V       |V                  |      | Same as the above except for <br> --lr 0.04
ResNet-56            | 8   |32  |V       |V                  |      | --target_model "resnet56_quant" --source_model "resnet56" --num_epochs 300 <br> --train_batch_size 128 <br> --test_batch_size 100 <br> --lr 0.04  <br> --lr_decay_steps [80, 120] <br> --lr_gamma 0.1 <br> --momentum 0.9  <br> --weight_decay 0.0001  <br> --lam 1  <br> --lam2 4  <br> --act_range 2  <br> --print_freq 200
ResNet-56            | 4 <br><br> 3 <br><br> 2   | 8 <br><br> 4 <br><br> 3  |V       |V                  |      |Same as the above except for <br> --lr 0.04 
DenseNet-40          | 8   |32  |V       |V                  |      | --target_model "densenet_40_quant" --source_model "densenet_40" --num_epochs 300 <br> --train_batch_size 128 <br> --test_batch_size 100 <br> --lr 0.1  <br> --lr_decay_steps [80, 120] <br> --lr_gamma 0.05 <br> --momentum 0.9  <br> --weight_decay 0.0001  <br> --lam 1  <br> --lam2 4  <br> --act_range 2  <br> --print_freq 200
DenseNet-40          | 4 <br><br> 3 <br><br> 2   | 8 <br><br> 4 <br><br> 3  |V       |V        |      | Same as the above except for <br> --lr 0.04

### Cross-domain tasks

#### DANN (VGG-2) on Digits (MNIST, MNISTM, SVHN, SynDigits).

Data                | Bit-width   | Pretrained Bit-width | DAQ    | Gradient Approx.  | CRP  | Hyperparameters 
---                 |:---:        |:---:                 |:---:   |:---:              |:---: |---                   
MNIST->MNISTM       | 4 <br><br> 3 <br><br> 2 <br><br> 1  |32     |V       |      |      | --model MNISTmodel_quant <br> --arch dann <br> --img_size 28 <br> --num_epochs 100 <br> --train_batch_size 64 <br> --test_batch_size 100 <br> --lr 0.0002  <br> --lr_decay_steps [80] <br> --lr_gamma 0.1 <br> --momentum 0.0  <br> --weight_decay 0.0  <br> --alpha 10  <br> --print_freq 600
MNIST->SVHN       | 4 <br><br> 3 <br><br> 2 <br><br> 1  |32     |V       |      |      | --model MNISTmodel_quant <br> --arch dann <br> --img_size 32 <br> --num_epochs 100 <br> --train_batch_size 100 <br> --test_batch_size 100 <br> --lr 0.01  <br> --lr_decay_steps [80] <br> --lr_gamma 0.1 <br> --momentum 0.9  <br> --weight_decay 0.0001  <br> --alpha 10  <br> --print_freq 500
SynDigits->MNIST       | 4 <br><br> 3 <br><br> 2 <br><br> 1  |32     |V       |      |      | --model MNISTmodel_quant <br> --arch dann <br> --img_size 28 <br> --num_epochs 128 <br> --train_batch_size 100 <br> --test_batch_size 100 <br> --lr 0.04  <br> --lr_decay_steps [80] <br> --lr_gamma 0.1 <br> --momentum 0.9  <br> --weight_decay 1e-6  <br> --alpha 10  <br> --print_freq 50

#### DANN (ResNet-50) on Office-31 (Amazon (A), DSLR (D), Webcam (W)).

The penalties μ and ρ in the ADMM optimization are set to 0.2 and 0.3, respectively

Data                | Bit-width   | Pretrained Bit-width | DAQ    | Gradient Approx.  | CRP  | Hyperparameters 
---                 |:---:        |:---:                 |:---:   |:---:              |:---: |---                   
A->W       | 8 <br><br> 5 <br><br> 4  |32     |V       |V      |V      | --model resnet50_dann <br> --arch resnet <br> --num_epochs 100 <br> --train_batch_size 32 <br> --test_batch_size 28 <br> --lr 0.001  <br> --lr_decay_steps [30, 60, 80] <br> --lr_gamma 0.1 <br> --momentum 0.9  <br> --weight_decay 0.0005  <br> --alpha 0  <br> --lam 1  <br> --lam2 4  <br> --act_range 2 <br> --print_freq 10
D->W       | 8 <br><br> 5 <br><br> 4  |32     |V       |V      |V      | --model resnet50_dann <br> --arch resnet <br> --num_epochs 100 <br> --train_batch_size 28 <br> --test_batch_size 28 <br> --lr 0.001  <br> --lr_decay_steps [30, 60, 80] <br> --lr_gamma 0.1 <br> --momentum 0.9  <br> --weight_decay 0.0005  <br> --alpha 0  <br> --lam 1  <br> --lam2 4  <br> --act_range 2 <br> --print_freq 10
W->D       | 8 <br><br> 5 <br><br> 4  |32     |V       |V      |V      | --model resnet50_dann <br> --arch resnet <br> --num_epochs 100 <br> --train_batch_size 32 <br> --test_batch_size 32 <br> --lr 0.001  <br> --lr_decay_steps [30, 60, 80] <br> --lr_gamma 0.1 <br> --momentum 0.9  <br> --weight_decay 0.0005  <br> --alpha 0  <br> --lam 1  <br> --lam2 4  <br> --act_range 2 <br> --print_freq 10
A->D       | 8 <br><br> 5 <br><br> 4  |32     |V       |V      |V      | --model resnet50_dann <br> --arch resnet <br> --num_epochs 100 <br> --train_batch_size 32 <br> --test_batch_size 32 <br> --lr 0.001  <br> --lr_decay_steps [30, 60, 80] <br> --lr_gamma 0.1 <br> --momentum 0.9  <br> --weight_decay 0.0005  <br> --alpha 0  <br> --lam 1  <br> --lam2 4  <br> --act_range 2 <br> --print_freq 10
D->A       | 8 <br><br> 5 <br><br> 4  |32     |V       |V      |V      | --model resnet50_dann <br> --arch resnet <br> --num_epochs 100 <br> --train_batch_size 32 <br> --test_batch_size 32 <br> --lr 0.001  <br> --lr_decay_steps [30, 60, 80] <br> --lr_gamma 0.1 <br> --momentum 0.9  <br> --weight_decay 0.0005  <br> --alpha 0  <br> --lam 1  <br> --lam2 4  <br> --act_range 2 <br> --print_freq 10
W->A       | 8 <br><br> 5 <br><br> 4  |32     |V       |V      |V      | --model resnet50_dann <br> --arch resnet <br> --num_epochs 100 <br> --train_batch_size 32 <br> --test_batch_size 32 <br> --lr 0.001  <br> --lr_decay_steps [30, 60, 80] <br> --lr_gamma 0.1 <br> --momentum 0.9  <br> --weight_decay 0.0005  <br> --alpha 0  <br> --lam 1  <br> --lam2 4  <br> --act_range 2 <br> --print_freq 10

#### DSAN (ResNet-50) on Office-31 (Amazon (A), DSLR (D), Webcam (W)).

DSAN on Office-31 has one more hyperparamter --param 0.3. The others' settings are same as DANN.
