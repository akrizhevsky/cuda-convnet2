# Introduction #

This page describes how to train one of the example networks on ImageNet. It presumes that you have already [compiled the code](Compiling.md) and [generated training data](Data.md).

# Training #

Now that the data batches are generated, we can train on them. Assuming you have compiled the code (see [Compiling](Compiling.md)), you can train on it by executing the following command (substituting your own paths where appropriate):

```
python convnet.py --data-path /usr/local/storage/akrizhevsky/ilsvrc-2012-batches --train-range 0-417 --test-range 1000-1016 --save-path /usr/local/storage/akrizhevsky/tmp  --epochs 90 --layer-def layers/layers-imagenet-1gpu.cfg --layer-params layers/layer-params-imagenet-1gpu.cfg --data-provider image --inner-size 224 --gpu 0 --mini 128 --test-freq 201 --color-noise 0.1
```

See [Arguments](Arguments.md) for a full listing of command-line arguments taken by `convnet.py`, but here's a brief description of the non-self-explanatory arguments passed here:

| **Argument** | **Meaning** |
|:-------------|:------------|
| --train-range 0-417 | train on data batches 0 .. 417 |
| --test-range 1000-1016 | test on data batches 1000 .. 1016 |
| --save-path /usr/local/storage/akrizhevsky/tmp | save checkpoints to given path |
| --epochs 90 | train for 90 iterations through the 418 batches |
| --gpu 0 | train on GPU device 0 |
| --mini 128 | train using SGD with minibatch of 128 examples |
| --test-freq 201 | compute the test error and save checkpoint every 201 training batches |
| --color-noise 0.1 | use [color channel noise](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) with standard deviation 0.1 |

The net will start producing output that looks like this:

```
1.0 (0.00%)... logprob:  7.156030, 1.000000, 0.995768 (8.428 sec)
1.1 (0.00%)... logprob:  7.064633, 0.999674, 0.996419 (8.364 sec)
1.2 (0.01%)... logprob:  7.001623, 0.999349, 0.994466 (8.365 sec)
```

which tells you the epoch number, the batch number, the training progress as a percentage, and the three values returned by the cost function. In our case the three values correspond to negative data log-probability, top-1 error rate, and top-5 error rate.

And then, when it saves a checkpoint, it also prints this:

```
======================Test output======================
logprob:  5.305025, 0.938802, 0.819987 
----------------------Averages-------------------------
logprob:  5.305025, 0.938802, 0.819987 
------------------------------------------------------- 
Layer 'conv1' weights[0]: 7.898808e-03 [2.681663e-05] [3.395022e-03] 
Layer 'conv1' biases: 2.679429e-04 [8.097122e-07] 
Layer 'conv2' weights[0]: 7.992361e-03 [4.403163e-06] [5.509215e-04] 
Layer 'conv2' biases: 9.996093e-01 [1.494067e-06] 
Layer 'conv3' weights[0]: 2.364471e-02 [2.431529e-06] [1.028360e-04] 
Layer 'conv3' biases: 7.078890e-04 [7.436417e-06] 
Layer 'conv4' weights[0]: 2.368718e-02 [3.326350e-06] [1.404283e-04] 
Layer 'conv4' biases: 9.992680e-01 [9.176imagenete-06] 
Layer 'conv5' weights[0]: 2.368802e-02 [4.521301e-06] [1.908687e-04] 
Layer 'conv5' biases: 9.988006e-01 [8.624080e-06] 
Layer 'fc4096a' weights[0]: 7.150869e-03 [1.230828e-06] [1.721229e-04] 
Layer 'fc4096a' biases: 9.999307e-01 [2.777610e-06] 
Layer 'fc4096b' weights[0]: 7.796859e-03 [3.070737e-06] [3.938428e-04] 
Layer 'fc4096b' biases: 9.987001e-01 [7.562206e-06] 
Layer 'fc1000' weights[0]: 7.961722e-03 [5.245239e-06] [6.588071e-04] 
Layer 'fc1000' biases: 6.999997e+00 [1.039037e-04] 
-------------------------------------------------------
Saved checkpoint to /usr/local/storage/akrizhevsky/tmp/ConvNet__2014-07-16_14.25.04
======================================================= (16.188 sec)
```

The line underneath "Test output" is the negative label log-probability, top-1 error rate, and top-5 error rate on one batch of the test set. The line underneath "Averages" is the average over all test batches.

The remaining lines indicate the scales of the weights and the weight updates. For example, in the line

```
Layer 'conv1' weights[0]: 7.898808e-03 [2.681663e-05] [3.395022e-03] 
```

the first number is the average absolute value in the weight matrix, the second number is the average absolute value in the weight increment matrix, and the third number is the ratio of the second to the first. Monitoring these quantities is helpful when debugging convergence problems.

## Stopping and resuming ##

If you want to change the learning parameters you specified, you can just kill the net with Ctrl-C (or kill -9, etc.), change the learning parameters in `layers/layer-params-imagenet-1gpu.cfg`, and then restart the net like this:

```
python convnet.py --load-file /usr/local/storage/akrizhevsky/tmp/ConvNet__2014-07-16_14.25.04
```

The net will resume from the last checkpoint.

## Generating features ##

If you want to generate feature maps from a layer, you can do it like this:

```
python convnet.py --load-file /usr/local/storage/akrizhevsky/tmp/ConvNet__2014-07-16_14.25.04 --write-features=fc1024 --feature-path/usr/local/storage/akrizhevsky/fc1024-features
```

This will write **test batch** features from the layer named `fc1024` to the path `/usr/local/storage/akrizhevsky/fc1024-features`. The features will be python pickled objects which you can open with `cPickle`.

## Computing the test error on a saved model ##

Given a checkpoint, you can compute the test error rate like this:

```
python convnet.py --load-file /usr/local/storage/akrizhevsky/tmp/ConvNet__2014-07-18_13.51.51 --test-only 1
```

Eventually this will produce output that looks like this:

```
======================Test output======================
logprob:  1.863741, 0.425480, 0.198100 
```

which tells you the error rate on the entire test set.

With the ImageNet data provider, you can also compute the test error rate averaged over 5 image patches and their horizontal reflections (see [this paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks), Section 4.1 for details):

```
python convnet.py --load-file /usr/local/storage/akrizhevsky/tmp/ConvNet__2014-07-18_13.51.51 --test-only 1 --multiview-test 1
```

which produces output:

```
======================Test output======================
logprob:  1.734335, 0.404200, 0.181500 
```

As you can see, the top-1 error rate has decreased by about 2%.

# CIFAR-10 example #

There's also a CIFAR-10 example config, and you can train on it like this:

```
python convnet.py --data-provider cifar --test-range 6 --train-range 1-5 --data-path /usr/local/storage/akrizhevsky/cifar-10-py-colmajor --inner-size 24 --save-path /usr/local/storage/akrizhevsky/ --gpu 0 --layer-def layers/layers-cifar10-11pct.cfg --layer-params layers/layer-params-cifar10-11pct.cfg
```

substituting your own [path to the CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar-10-py-colmajor.tar.gz).