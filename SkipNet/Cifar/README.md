
# Data Preparation and Models

We have used pretrained SkipNet SP model with ResNet 110. For data preparation and pretrained models, we followed [this](https://github.com/ucbdrive/skipnet/tree/master/cifar).

# How to Run
```
 python2 ILFO_Cifar_Skip.py test cifar10_rnn_gate_110 -d cifar10
 
```
# Output

Create a folder output. Input/Output images will be saved there as img_{batch_no}_{img_no}_{orig/ILFO}_{inc_in_FLOPs}.png.
