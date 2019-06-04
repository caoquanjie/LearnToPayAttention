# LearnToPayAttention
a tensorflow implementation of ICLR 2018 paper Learn To Pay Attention: [https://arxiv.org/pdf/1804.02391](https://arxiv.org/pdf/1804.02391)

I implemented only one version: VGG-att-concat-dp, and I trained the model on CIFAR-10, CIFAR-100 DATASET.
Finally, I use the pretrained CIFAR-100 model initialise the weights in CUB finetune.

### Requirements
python 3.6 </br>
tensorflow 1.4.0 </br>
numpy 1.12.0 </br>
skimage

### Training 
#### 1. In cifar10_attention 
Run `python train.py --batch_size=64 --total_step=100000 --result_log='att.log'`
#### 2. In cifar100_attention
Run `python train.py --batch_size=64 --total_step=200000 --result_log='att.log'`
#### 3. In CUB_finetune
Run `python checkpoint_to_npy.py` to store the model of CIFAR100 dataset in *.npy* format.</br>
Run `python dataset_to_tfrecords.py` to get `train.tfrecords` and `test.tfrecords` of CUB-200-2011. Source data can be downloaded in [Caltech-UCSD Webpage](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).</br>
Run `python main.py --batch_size=32 --learning_rate_start=1.6 --learning_rate_decay=0.5 --total_step=200000 --checkpoint_dir='./models/'

## Results
after 100000 steps, the accuracy with VGG-att2-concat-dp is reached `94.79%` in CIFAR10 dataset.</br>
after 200000 steps, the accuracy with VGG-att2-concat-dp is reached `77.64%` in CIFAR100 dataset.</br> 
after 100000 steps, the finetune accuracy with VGG-att3-concat-dp is reached `73.25%` in CUB-200-2011.</br>
 
### Attention map visualization (on test data of CIFAR-10)

![](https://github.com/caoquanjie/LearnToPayAttention/raw/master/images/fig.jpg)
 