# TransformerVAE for Sketch Tutoring
This repository contains the code for the conference paper "Variational-autoencoder-based Environment for Interactive Sketch Tutoring Aiming for Kids", accepted by [WAIE 2020](https://dl.acm.org/doi/abs/10.1145/3447490.3447493?fbclid=IwAR2EujIfLIvzsduNHkD-4yMyQGRPnqKKVTr3IojzfPrvORKTeVOc38clMmc)



![image](https://github.com/shanjgit/TransformerVAE-for-Sketch-Tutoring/blob/master/misc/TransformerVAE_demo.gif)

Inspired by the generative model [sketch-rnn](https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html) proposed by David Ha et. al. We train a VAE model using [Google QuickDraw dataset](https://quickdraw.withgoogle.com/data). Besides using CNN-based encoder to encode space information, we further replaced the the auto-regressinve network by a transformer decoder to enhence the stability of the generated drawing in an interative environment.

## Requirement
Python >= 3.6
tensorflow >= 2.0










