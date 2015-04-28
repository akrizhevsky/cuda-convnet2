<h1>Table of Contents</h1>


# Parallelism #

`cuda-convnet2` supports three ways to parallelize the training of your model across multiple GPUs inside one machine. They are:

  * Data prallelism
  * Model parallelism
  * The hybrid approach described in [[1](http://arxiv.org/abs/1404.5997)].

All three approaches are described in [[1](http://arxiv.org/abs/1404.5997)], so the focus here is not to repeat the explanations, but to show how to implement the schemes in `cuda-convnet2`.

## A note about GPU indices ##

In the examples below, layers are assigned to GPUs using the syntax
```
gpu=0
```

Please note that this does not actually refer to the 0 device returned by `nvidia-smi`, but rather to the first argument passed on the command-line `--gpu` option. In other words, if you run `cuda-convnet2` with the line

```
python convnet.py --load-file /path/to/checkpoint --gpu 3,2,1,0
```

then `gpu=0` will refer to CUDA device 3.

## Data parallelism ##

Data parallelism is the simplest parallelization scheme, and it's also the simplest to specify in `cuda-convnet2`. In data parallelism, each GPU is given different data examples to train on, and the GPUs synchronously train the same model on their respective data examples. You can specify a data-parallel layer in a `cuda-convnet2` configuration file like this:

```
[conv1]
type=conv
inputs=data
channels=3
filters=64
padding=0
stride=4
filterSize=11
initW=0.01
sumWidth=4
sharedBiases=1
gpu=0,1
```

This indicates that this layer should run on two GPUs, and each GPU will be given half of each minibatch to train on. All subsequent layers inherit the GPUs of their first input layer, so it is not necessary to add `gpu=0,1` to the remaining layer definitions.

The layer `conv1` is said to consist of two **replicas**, because it's the same layer running on each GPU, just processing different data.

**Note that since each GPU will now be running only half of each minibatch, you should increase the minibatch size to preserve computational efficiency. In other words, you'll want to run this model with a minibatch size of 256 instead of 128**.

```
python convnet.py --data-path /usr/local/storage/akrizhevsky/ilsvrc-2012-batches --train-range 0-417 --test-range 1000-1016 --save-path /usr/local/storage/akrizhevsky/tmp  --epochs 90 --layer-def layers/layers-imagenet-2gpu-data.cfg --layer-params layers/layer-params-imagenet-2gpu-data.cfg --data-provider image --inner-size 224 --gpu 0 --mini 256 --test-freq 201 --color-noise 0.1
```

This will train the model with an effective minibatch size of 256, with each GPU training on 128 examples at a time.

## Model parallelism ##

In model parallelism, different GPUs train different layers on the same data examples. Specifying this kind of parallelism is unfortunately a bit more verbose in the config file, but it isn’t hard:

```
[conv1a]
type=conv
inputs=data
channels=3
filters=64
padding=0
stride=4
filterSize=11
initW=0.01
sumWidth=4
sharedBiases=1
gpu=0

[conv1b]
type=conv
inputs=data
channels=3
filters=64
padding=0
stride=4
filterSize=11
initW=0.01
sumWidth=4
sharedBiases=1
gpu=1
```

This says that the layer conv1a will run on GPU 0 while the layer conv1b will run on GPU 1. The layers take the same input data and execute in parallel.

Any layer can take any other layer as input, including layers that run on different GPUs. For example:

```
[conv1a]
type=conv
inputs=data
channels=3
filters=64
padding=0
stride=4
filterSize=11
initW=0.01
sumWidth=4
sharedBiases=1
gpu=0

[conv1b]
type=conv
inputs=data
channels=3
filters=64
padding=0
stride=4
filterSize=11
initW=0.01
sumWidth=4
sharedBiases=1
gpu=1

[conv2a]
type=conv
inputs=conv1a,conv1b
channels=64
filters=128
padding=0
stride=1
filterSize=3
initW=0.01
sumWidth=4
sharedBiases=1
gpu=0

[conv2b]
type=conv
inputs=conv1a,conv1b
channels=64
filters=128
padding=0
stride=1
filterSize=3
initW=0.01
sumWidth=4
sharedBiases=1
gpu=1
```

Here, the layers conv2a and conv2b take both conv1a and conv1b as inputs. An implicit copy operation is performed in order to get the output of conv1a into the input of conv2b, as well as the output of conv1b into the input of conv2a.

If you have more than two GPUs, you can also use data parallelism and model parallelism simultaneously:

```
[conv1a]
type=conv
inputs=data
channels=3
filters=64
padding=0
stride=4
filterSize=11
initW=0.01
sumWidth=4
sharedBiases=1
gpu=0,1

[conv1b]
type=conv
inputs=data
channels=3
filters=64
padding=0
stride=4
filterSize=11
initW=0.01
sumWidth=4
sharedBiases=1
gpu=2,3

[conv2a]
type=conv
inputs=conv1a
channels=64
filters=128
padding=0
stride=1
filterSize=3
initW=0.01
sumWidth=4
sharedBiases=1
gpu=0,1

[conv2b]
type=conv
inputs=conv1b
channels=64
filters=128
padding=0
stride=1
filterSize=3
initW=0.01
sumWidth=4
sharedBiases=1
gpu=2,3
```

This implements a "two-tower" model in which there are two replicas of each tower, where each replica trains on different data examples. The first tower, consisting of conv1a and conv2a, runs on GPUs 0 and 1, while the second tower, consisting of conv1b and conv2b, runs on GPUs 2 and 3.

**Note**: in general, cost layers cannot be partitioned across multiple GPUs in this way. Model parallelism cannot be applied there in `cuda-convnet2`. See [this model-parallel config file](https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-2gpu-model.cfg) for an example.

## The hybrid approach described in `[1]` ##

As argued in [[1](http://arxiv.org/abs/1404.5997)], it can be beneficial to use different types of parallelism in different layers. In particular, it's nice to be able to use data parallelism in the convolutional layers and model parallelism in the fully-connected layers. Here's how you do it:

```
[pool3]
type=pool
pool=max
inputs=conv5
sizeX=3
stride=2
channels=256
neuron=relu
gpu=0,1

[fc2048a]
type=fc
inputs=pool3
outputs=2048
initW=0.01
initB=1
neuron=relu
gpu=0

[fc2048b]
type=fc
inputs=pool3
outputs=2048
initW=0.01
initB=1
neuron=relu
gpu=1
```

So the syntax is just a composition of the syntaxes for data and model parallelism. The layer `pool3` here has two replicas while the following layers each have one replica.

**Note:** there are various restrictions on how you can compose data parallelism and model parallelism in `cuda-convnet2`, mainly due to my laziness. In particular,

  * For a given layer, all layers previous of it must have equal number of replicas.
  * For a given layer, all layers next of it must have equal number of replicas.
  * The number of replicas in a given layer must be divisible by the number of replicas in every layer next of it.

As further argued in [[1](http://arxiv.org/abs/1404.5997)], it also can be beneficial to update the weights of the fully-connected layers more often than the weights of the convolutional layers.

To understand the protocol for specifying this in `cuda-convnet2`, you have to understand a bit about how this form of parallelism is implemented. As depicted in Figure 2 of [[1](http://arxiv.org/abs/1404.5997)], there is no longer one forward and one backward pass per minibatch. Instead, there are multiple little forward passes through the fully-connected layers. Specifically, the number of forward (equivalently, backward) passes is given by

![https://chart.googleapis.com/chart?cht=tx&chl=\frac{\textrm{maximum%20number%20of%20replicas}}{\textrm{minimum%20number%20of%20replicas}}&nonsense=something.png](https://chart.googleapis.com/chart?cht=tx&chl=\frac{\textrm{maximum%20number%20of%20replicas}}{\textrm{minimum%20number%20of%20replicas}}&nonsense=something.png)

In our example the maximum number of replicas in a layer is two (`conv1` has two replicas) while the minimum is one (`fc2048a` and `fc2048b` have one replica each). So each forward propagation of a minibatch will consist of two forward passes and two backward passes. Referring to Figure 2 of [[1](http://arxiv.org/abs/1404.5997)] again, the first forward pass propagates through the whole net, while the second forward pass propagates only through the fully-connected layers.

Since there are two backward passes through the fully-connected layers, we can either
  * accumulate the gradient throughout the passes and update the fully-connected weights after the final backward pass, or
  * update the fully-connected weights during each backward pass.

This is controlled with the `updatePeriod` parameter in the layer parameter file. For example, to update the fully-connected weights as often as possible, you can write:

```
[fc2048a]
momW=0.9
momB=0.9
wc=0.0005
wball=0
epsW=dexp[base=0.01;tgtFactor=250;numSteps=4]
epsB=dexp[base=0.02;tgtFactor=10;numSteps=2]
updatePeriod=1
```

`updatePeriod=1` indicates that the net should update the weights during every backward pass. Setting `updatePeriod=2` would indicate that the net should update the weights every two backward passes (accumulating the gradient throughout). And so forth. Obviously, the `updatePeriod` must divide the total number of backward passes per minibatch.

## Quick multi-GPU examples ##

Here are some examples of multi-GPU config files and run-lines to give you an idea how to run these models.

### Two-GPU data parallelism ###
```
python convnet.py --data-path /usr/local/storage/akrizhevsky/ilsvrc-2012-batches --train-range 0-417 --test-range 1000-1016 --save-path /usr/local/storage/akrizhevsky/tmp  --epochs 90 --layer-def layers/layers-imagenet-2gpu-data.cfg --layer-params layers/layer-params-imagenet-2gpu-data.cfg --data-provider image --inner-size 224 --gpu 0,1 --mini 256 --test-freq 201 --color-noise 0.1
```

### Two-GPU model parallelism ###
Note: this is not exactly the same model as above.
```
python convnet.py --data-path /usr/local/storage/akrizhevsky/ilsvrc-2012-batches --train-range 0-417 --test-range 1000-1016 --save-path /usr/local/storage/akrizhevsky/tmp  --epochs 90 --layer-def layers/layers-imagenet-2gpu-model.cfg --layer-params layers/layer-params-imagenet-2gpu-model.cfg --data-provider image --inner-size 224 --gpu 0,1 --mini 128 --test-freq 201 --color-noise 0.1
```

### Four-GPU data parallelism ###
```
python convnet.py --data-path /usr/local/storage/akrizhevsky/ilsvrc-2012-batches --train-range 0-417 --test-range 1000-1016 --save-path /usr/local/storage/akrizhevsky/tmp  --epochs 90 --layer-def layers/layers-imagenet-4gpu-data.cfg  --layer-params layers/layer-params-imagenet-4gpu-data.cfg --data-provider image --inner-size 224 --gpu 0,1,2,3 --mini 512 --test-freq 201 --color-noise 0.1
```

### Four-GPU data parallelism in convolutional layers, model parallelism in fully-connected layers ###
```
python convnet.py --data-path /usr/local/storage/akrizhevsky/ilsvrc-2012-batches --train-range 0-417 --test-range 1000-1016 --save-path /usr/local/storage/akrizhevsky/tmp  --epochs 90 --layer-def layers/layers-imagenet-4gpu-data-model.cfg --layer-params layers/layer-params-imagenet-4gpu-data-model.cfg --data-provider image --inner-size 224 --gpu 0,1,2,3 --mini 512 --test-freq 201 --color-noise 0.1
```

## References ##
  1. [One weird trick for parallelizing convolutional neural networks](http://arxiv.org/abs/1404.5997)