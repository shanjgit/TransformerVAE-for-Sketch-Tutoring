# TransformerVAE for Sketch Tutoring
This repository contains the code for the conference paper "Variational-autoencoder-based Environment for Interactive Sketch Tutoring Aiming for Kids", accepted by [WAIE 2020](https://dl.acm.org/doi/abs/10.1145/3447490.3447493?fbclid=IwAR2EujIfLIvzsduNHkD-4yMyQGRPnqKKVTr3IojzfPrvORKTeVOc38clMmc)



![image](https://github.com/shanjgit/TransformerVAE-for-Sketch-Tutoring/blob/master/misc/TransformerVAE_demo.gif)

Inspired by the generative model [sketch-rnn](https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html) proposed by David Ha et. al. We trained a VAE model using [Google QuickDraw dataset](https://quickdraw.withgoogle.com/data). Besides using CNN-based encoder to encode space information, we further replaced the the auto-regressinve network by a transformer decoder to enhence the stability of the generated drawing in an interative environment. Python code is modified from [this repository](https://github.com/eyalzk/sketch_rnn_keras). The tfjs code and GUI is modified from the open source code in [this repository](https://github.com/magenta/magenta-demos/tree/main/sketch-rnn-js)

## Requirements
Python >= 3.6  
tensorflow >= 2.0  

## Weight Converting
1. Using Python code to train and save model weights:   
```
python new_seq2seqVAE_train.py  
```

2. Converting the weights into `tfjs` format:  
```
# bash

tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir  
```

3. The model wieghts can then be initilized in tfjs  
```

```







