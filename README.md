# LearnToPayAttention
## a tensorflow implementation of ICLR 2018 paper Learn To Pay Attention

I implemented two versions: VGG-att1-concat-dp and VGG-att2-concat-dp, and I trained the model on CIFAR-10, CIFAR-100 DATASET.
Finally, we use the pretrained CIFAR-100 model initialise the weights in CUB finetune.

### requirements
python 3.6 </br>
tensorflow 1.4.0 </br>
numpy 1.12.0 </br>
skimage

### training 
#### 1. in cifar10_attention
python train.py
#### 2. in cifar100_attention
python train.py
#### 3. in CUB_finetune
python main.py

## results
after 100000 steps, the accuracy with VGG-att2-concat-dp is reached 94.79% in CIFAR10 dataset.</br>
after 200000 steps, the accuracy with VGG-att2-concat-dp is reached 77.64% in CIFAR100 dataset.</br> 
after 100000 steps, the finetune accuracy with VGG-att2-concat-dp is reached 73.25% in CUB-200-2011.</br>
 
### Attention map visualization (on test data of CIFAR-10)
