<h1>Table of Contents</h1>


# Introduction #

To define the architecture of your neural net, you must write a layer definition file. You can find several complete and working layer definition files [here](https://code.google.com/p/cuda-convnet2/source/browse/#git%2Flayers). In this section I will go over the types of layers supported by the net, and how to specify them.

# Layer definition file - basic features #

## Data layer ##

The first thing you might add to your layer definition file is a data layer.

```
[data]
type=data
dataIdx=0

[labels]
type=data
dataIdx=1

# By the way, this is a comment.
```

The things in square brackets are user-friendly layer names, and they can be whatever you want them to be. Here we're really defining two layers: one we're calling **data** and the other we're calling **labels**.

The **type=data** line indicates that this is a data layer.

Our python [data provider](https://code.google.com/p/cuda-convnet2/source/browse/convdata.py#198) outputs a list of three elements: the images, the labels, and the labels in one-hot encoded form. The line **dataIdx=0** indicates that the layer named **data** is mapped to the images, and similarly the line **dataIdx=1** indicates that the layer named **labels** is mapped to the labels.

## Convolution layer ##

Convolution layers apply a small set of filters all over their input "images". They're specified like this:

```
[conv32]
type=conv
inputs=data
channels=3
filters=32
padding=4
stride=1
filterSize=9
neuron=logistic
initW=0.00001
initB=0.5
sharedBiases=true
sumWidth=4
```

Again, the bracketed **conv32** is the name we're choosing to give to this layer.

Here's what the other parameters mean:
| **Parameter** | **Default value** | **Meaning** |
|:--------------|:------------------|:------------|
|`type=conv`|`--`|defines this as a convolution layer|
|`inputs=data` |`--`|says that this layer will take as input the layer named **data** (defined above) |
|`channels=3` |`--`| tells the net that the layer named **data** produces 3-channel images (i.e. color images). Since the images are assumed to be square, that is all that you have to tell it about the data dimensionality. This value must be either 1, 2, 3, or a multiple of 4.|
| `filters=32`|`--`| says that this layer will apply 32 filters to the images. This number must be a multiple of 16. |
|`padding=4` |`0`| instructs the net to implicitly pad the images with a 4-pixel border of zeros (this does not cause it to create a copy of the data or use any extra memory). Set to 0 if you don't want padding.|
|`stride=1` |`1`|indicates that the distance between successive filter applications should be 1 pixel. |
|`filterSize=9` |`--`|says that this layer will use filters of size 9x9 pixels (with 3 channels). |
|`neuron=logistic` |`ident`| defines the neuron activation function to be applied to the output of the convolution. If omitted, no function is applied. See NeuronTypes for supported types. **All layers except data and cost layers can take a neuron parameter.**|
|`initW=0.00001` |`0.01`| instructs the net to initialize the weights in this layer from a normal distribution with mean zero and standard deviation 0.00001.|
|`initB=0.5` |`0`| instructs the net to initialize the biases in this layer to 0.5.|
|`sharedBiases=true` |`true`| indicates that the biases of every filter in this layer should be shared amongst all applications of that filter (which is how convnets are usually trained). Setting this to false will untie the biases, yielding a separate bias for every location at which the filter is applied. |
|`sumWidth=4` |`--`| this is a parameter that affects the performance of the weight gradient computation. It's a bit hard to predict what value will result in the best performance (it's problem-specific), but it's worth trying a few. Valid values are ones that are smaller than the width (equivalently, length) of the output grid in this convolutional layer. For example if this layer produces 32-channel 20x20 output grid, valid values of `sumWidth` are integers in the range [1,20]. This parameter also has an impact on memory consumption. The amount of extra memory used will be: `(number of weights in this layer)x(number of outputs this layer produces) / sumWidth^2.` |

This convolutional layer will produce 32 x _N_ x _N_ output values, where _N_ is the number of filter applications required to cover the entire width (equivalently, height) of the image plus zero-padding with the given stride. As mentioned above, each filter in this layer has 3x9x9 weights.

The parameters with default values may be omitted from the configuration file.

### Performance notes ###
  * The computation will be most efficient when `filters` and `channels` are divisible by 64.

## Locally-connected layer with unshared weights ##

This kind of layer is just like a convolutional layer, but without any weight-sharing. That is to say, a different set of filters is applied at every (x, y) location in the input image. Aside from that, it behaves exactly as a convolutional layer.

Here's how to define such a layer, taking the **conv32** layer as input:

```
[local32]
type=local
inputs=conv32
channels=32
filters=32
padding=4
stride=1
filterSize=9
neuron=logistic
initW=0.00001
initB=0
```

Aside from the **type=local** line, there's nothing new here. Note however that since there is no weight sharing, the actual number of distinct filters in this layer will be `filters` multiplied by however many filter applications are required to cover the entire image and padding with the given stride.

### Performance notes ###
  * The computation will be most efficient when `filters` and `channels` are divisible by 64.

## Fully-connected layer ##

A fully-connected layer simply multiplies its input by a weight matrix. It's specified like this:

```
[fc1024]
type=fc
outputs=1024
inputs=data
initW=0.001
neuron=relu
```

The only parameter here that we have not yet seen is **outputs=1024**. It does what you would expect -- it indicates that this layer should have 1024 neurons.

## Local pooling layer ##

This type of layer performs local, **per-channel** pooling on its input.

```
[maxpool]
type=pool
pool=max
inputs=local32
start=0
sizeX=4
stride=2
outputsX=0
channels=32
neuron=relu
```

A few new parameters here:
| **Parameter** | **Default value** | **Meaning** |
|:--------------|:------------------|:------------|
|`pool=max` |`--`|indicates that this is to be a max-pooling layer. Also supported is **pool=avg** for average-pooling. |
|`inputs=conv32` |`--`|indicates that this layer subsamples the layer named **conv32**.|
|`start=0` |`0`|tells the net where in the input image to start the pooling (in x,y coordinates). In principle, you can start anywhere you want. Setting this to a positive number will cause the net to discard some pixels at the top and at the left of the image. Setting this to a negative number will cause it to include pixels that don't exist (which is fine). **start=0** is the usual setting. |
|`sizeX=4` |`--`|defines the size of the pooling region in the x (equivalently, y) dimension. Squares of size (**sizeX**)<sup>2</sup> get reduced to one value by this layer. There are no restrictions on the value of this parameter. It's fine for a pooling square to fall off the boundary of the image.|
|`stride=2` |`--`|defines the stride size between successive pooling squares. Setting this parameter smaller than **sizeX** produces _overlapping_ pools. Setting it equal to **sizeX** gives the usual, non-overlapping pools. Values greater than **sizeX** are not allowed. |
|`outputsX=0` |`0`|allows you to control how many output values in the x (equivalently, y) dimension this operation will produce. This parameter is analogous to the **start** parameter, in that it allows you to discard some portion of the image by setting it to a value small enough to leave part of the image uncovered. Setting it to zero instructs the net to produce as many outputs as is necessary to ensure that the whole image is covered. |

Since the pooling performed by this layer is per-channel, the number of output channels is equivalent to the number of input channels. But the size of the image in the x,y dimensions gets reduced.

## Local response normalization layer (same map) ##

This kind of layer computes the function

![http://cuda-convnet.googlecode.com/svn/wiki/images/rnorm.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/rnorm.gif)

where ![http://cuda-convnet.googlecode.com/svn/wiki/images/u_fxy.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/u_fxy.gif) is the activity of a unit in map _f_ at position _x,y_ prior to normalization, _S_ is the image size, and _N_ is the size of the region to use for normalization. The output dimensionality of this layer is always equal to the input dimensionality.

This type of layer turns out to be useful when using neurons with unbounded activations (e.g. rectified linear neurons), because it permits the detection of high-frequency features with a big neuron response, while damping responses that are uniformly large in a local neighborhood. It is a type of regularizer that encourages "competition" for big activities among nearby groups of neurons.

Here's how this layer is specified:

```
[rnorm1]
type=rnorm
inputs=maxpool
channels=32
size=5
```

| **Parameter** | **Meaning** |
|:--------------|:------------|
| `channels=32` | indicates that this layer takes 32-channel input because that's what the **maxpool** layer produces. The number of "channels" here just serves to define the shape of the input and has no actual bearing on the output (unlike in convolutional layers, which sum over channels). |
| `size=5` | the _N_ variable in the above formula, this defines the size of the local regions used for response normalization. Squares of (**size**)<sup>2</sup> are used to normalize each pixel. The squares are centered at the pixel. |

The _alpha_ and _beta_ variables are specified in the [layer parameter file](LayerParams#Local_response/contrast_normalization_layers.md), so that you can change them during training.

## Local response normalization layer (across maps) ##

This layer is just like the one described above, but units are divided only by the activities of other units **in the same position but in different maps (channels)**.

It's specified like this:

```
[rnorm2]
type=cmrnorm
inputs=maxpool
channels=32
size=5
```

The **size** parameter indicates how many nearby maps to use for normalization. For example, if your layer has 32 maps, a unit in the 7th map will be normalized by units in the 5th through 9th maps when **size** is set to 5.

Specifically, the function that this layer computes is:

![http://cuda-convnet.googlecode.com/svn/wiki/images/rnorm-crossmap.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/rnorm-crossmap.gif)

Here _F_ is the number of maps. The _alpha_ and _beta_ parameters are specified in the layer parameter file, described here: [LayerParams#Layer\_parameter\_file\_-\_basic\_features](LayerParams#Layer_parameter_file_-_basic_features.md)

## Local contrast normalization layer ##

This kind of layer computes the function

![http://cuda-convnet.googlecode.com/svn/wiki/images/cnorm.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/cnorm.gif)

![http://cuda-convnet.googlecode.com/svn/wiki/images/m_fxy.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/m_fxy.gif) here is the mean of all ![http://cuda-convnet.googlecode.com/svn/wiki/images/u_fxy.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/u_fxy.gif) in the 2D neighbourhood defined by the above summation bounds. This layer is very similar computationally to response normalization -- the difference being that the denominator here computes the variance of activities in each neighborhood, rather than just the sum of squares (correlation).

Here's how this layer is specified:

```
[cnorm1]
type=cnorm
inputs=rnorm1
channels=32
sizeX=7
scale=0.001
pow=0.5
```

The meanings of all of these parameters are the same as in the response normalization layer described above.

## Neuron layer ##

This layer takes one layer as input and applies a neuron activation function to it. See NeuronTypes for available activation functions.

This layer is specified like this:

```
[neuron1]
type=neuron
inputs=layer1
neuron=logistic
```

Note that all layers except data layers and cost layers can take a **neuron=x** parameter, so there is often no need to explicitly define these neuron layers.

Side-note: this layer is admittedly misnamed. It just applies an element-wise function to its input. So don't think of neurons with weights. This layer has no weights.

## Elementwise sum layer ##

This kind of layer simply performs an elementwise weighted sum of its input layers. Of course this implies that all of its input layers must have the same dimensionality. It's specified like this:

```
[sumlayer]
type=eltsum
inputs=layer1,layer2,layer3
coeffs=0.5,-1,1.2
```

This layer will produce the weighted sum 0.5 x **layer1** - **layer2** + 1.2 x **layer3**.

| **Parameter** | **Default value** | **Meaning** |
|:--------------|:------------------|:------------|
|`coeffs=0.5,-1,1.2` |`1,...`|the coefficients of summation for the specified inputs.|

The number of output values in this layer is of course equal to the number of output values in any of the input layers.

## Elementwise max layer ##

This kind of layer performs an elementwise max of its input layers.

It is specified like this:

```
[maxlayer]
type=eltmax
inputs=layer1,layer2,layer3
```

This will produce an output matrix in which every element at position (_i_,_j_) is the max of the (_i_,_j_)th elements of **layer1**, **layer2**, and **layer3**.

## Dropout layer ##

This layer implements [dropout](http://www.cs.toronto.edu/~hinton/absps/dropout.pdf).

It is specified like this:

```
[dropout1]
type=dropout
inputs=fc4096a
```

The probability of dropping out a neuron is given in the [layer parameter file](https://code.google.com/p/cuda-convnet2/wiki/LayerParams#Dropout_layer_parameters) (because you can change it during training).

## Softmax layer ##

This layer computes the function

![http://cuda-convnet.googlecode.com/svn/wiki/images/softmax.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/softmax.gif)

where each _x<sub>i</sub>_ is an input value. It's specified like this:

```
[fc10]
type=fc
outputs=10
inputs=conv32
initW=0.0001

[probs]
type=softmax
inputs=fc10
```

Here we're really defining two layers -- **fc10** is a fully-connected layer with 10 output values (because our dataset has 10 classes), and **probs** is a softmax that takes **fc10** as input and produces 10 values which can be interpreted as probabilities. This type of layer is useful for classification tasks.

## Concatenation layer ##

This kind of layer concatenates its input layers:

```
[concat]
type=concat
inputs=layer1,layer2,layer3
```

The number of neurons in this layer is equal to the sum of the number of neurons in all input layers.

There is a slightly more advanced form of concatenation layer, called a pass-through layer:

```
[passcat]
type=pass
inputs=layer1,layer2,layer3
```

This type of layer is semantically equivalent to a concatenation layer, but it does not actually require any extra memory, and it does not cause a copy operation to be performed. Instead, it imposes a constraint on the memory regions occupied by its input layers `layer1`, `layer2`, and `layer3`. This layer forces the neuron activations of its input layers to occupy a contiguous chunk of memory. So no copy operation is needed in order to get the concatenation. Obviously there are constraints on where these types of layers can go, and it is easy to construct examples with multiple pass-through layers that lead to incoherent constraints. I think the code is able to catch these cases, though, so you shouldn't get into much trouble using them.

## Logistic regression cost layer ##

A net must define an objective function to optimize. Here we're defining a (multinomial) [logistic regression](http://en.wikipedia.org/wiki/Multinomial_logistic_regression) objective. It's specified like this:

```
[logprob]
type=cost.logreg
inputs=labels,probs
```

The **cost.logreg** objective takes two inputs -- true labels and predicted probabilities. We defined the **labels** layer early on, and the **probs** layer just above.

## Binomial cross-entropy cost layer ##

This objective minimizes the sum of the cross-entropies between each pair of input neurons:

```
[bce]
type=cost.bce
inputs=labmat,probs
```

Specifically, it computes

![https://chart.googleapis.com/chart?cht=tx&chl=\sum_{i}{p%28x_i%29\log%28q%28x_i%29%29%2B%281-p%28x_i%29%29\log%281-q%28x_i%29%29}&param=.png](https://chart.googleapis.com/chart?cht=tx&chl=\sum_{i}{p%28x_i%29\log%28q%28x_i%29%29%2B%281-p%28x_i%29%29\log%281-q%28x_i%29%29}&param=.png)

where _i_ indexes over the input dimensionality (`labmat` and `probs` must have equal dimensionality), _p(x)_ refers to `labmat` and _q(x)_ refers to `probs`.

One scenario in which using this kind of cost makes sense is with binary labels and logistic predictions. If you use the `image` data provider that the example layers use, you can optimize a net with binomial cross-entropy cost this way:

```
[labmat]
type=data
dataIdx=2

...

[fc1000]
type=fc
outputs=1000
inputs=pass2
initW=0.01
initB=-7
neuron=logistic

[bce]
type=cost.bce
inputs=labmat,fc1000
```

`dataIdx=2` in the layer `labmat` indicates that we want the 3rd matrix returned by the data provider. For the `image` data provider, this happens to be one-hot encoded a matrix with dimensionality equal to the number of labels.

## Sum-of-squares cost layer ##

This objective minimizes the squared L2 norm of the layer below. One use for it is to train autoencoders. For example, you can minimize reconstruction error like this:

```
[diff]
inputs=recon,data
type=eltsum
coeffs=1,-1

[sqdiff]
type=cost.sum2
inputs=diff
```

This defines two layers -- the first subtracts the "data" from the "reconstruction" and the second is the sum-of-squares cost layer which minimizes the sum of the squares of its input.

# Layer definition file - data manipulation layers #

Data manipulation layers pre-process the data in some way. Normal use cases do not involve propagating gradient through them, and some of them do not support gradient propagation.

## Gaussian blur layer ##

This kind of layer is useful if you're interested in multi-scale learning. Many vision systems learn (or simply apply) features at multiple scales. To do that, it's good to be able to subsample the input images in a way that doesn't introduce a lot of aliasing. Blurring them achieves that.

So here's how you specify a [Gaussian blur](http://en.wikipedia.org/wiki/Gaussian_blur) layer:

```
[myblur]
type=blur
inputs=data
filterSize=5
stdev=2
channels=3
```

The non-obvious parameters are:

| **Parameter** | **Default value** | **Meaning** |
|:--------------|:------------------|:------------|
|`filterSize=5` |`--`|the width (in pixels) of the Gaussian blur filter to apply to the image. This must be one of 3, 5, 7, or 9.|
|`stdev=1` |`--`|the standard deviation of the Gaussian. This can be any positive float.|

The number of output values in this layer is equal to the number of input values.

The layer defined above will convolve its input layer **data** with the following 1-D filter (both horizontally and vertically, of course):
```
[0.05448869,  0.24420136,  0.40261996,  0.24420136,  0.05448869]
```

Note that no matter what parameters you specify, the sum of all elements in the filter will always be 1. Intuitively, wider Gaussians (i.e. bigger standard deviations) require an increase in the `filterSize` parameter.

## Bed of nails layer ##

This layer applies a [Dirac comb](http://en.wikipedia.org/wiki/Shah_function) to the input image. Use this layer to subsample a Gaussian-blurred image.

It's specified like this:

```
[mynails]
type=nailbed
inputs=myblur
stride=2
channels=3
```

The only new parameter here is `stride`, which indicates the spacing (in pixels) between samples. With a stride of 2, every other pixel is skipped, so the output image will be roughly 4 times smaller in area.

## Image resizing layer ##

Gaussian blur + bed-of-nails subsampling can reduce your image size by an integer factor, but you might also want to resize your images by a smaller amount. This layer implements image resizing with [bilinear filtering](http://en.wikipedia.org/wiki/Bilinear_filtering). It can cleanly (i.e. without aliasing) resize images by a factors roughly within the range 0.5-2.

It's specified like this:

```
[resiz]
type=resize
inputs=data
scale=1.3
channels=3
```

This will scale the input images _down_ by a factor of 1.3. So if your input consisted of 32x32 images, the output of this layer will have dimensions 24x24 (it rounds down from 32/1.3 = 24.6).

**Note**: This layer cannot propagate gradient. So it should only be placed over layers that do not need to compute the gradient (e.g. the data layer).

## Color space manipulation layers ##

These layers convert between color spaces. Note that they cannot propagate gradient, and as such must not be placed over any layers with weights if you expect to learn those weights.

### RGB to YUV layer ###

This layer converts RGB-coded images to YUV-coded images.

Specifically, it applies the following linear transformation to the color channels of the input:

![http://cuda-convnet.googlecode.com/svn/wiki/images/rgb_to_yuv.png](http://cuda-convnet.googlecode.com/svn/wiki/images/rgb_to_yuv.png)

This layer is specified like this:

```
[yuv]
type=rgb2yuv
inputs=data
```

### RGB to L`*`a`*`b`*` layer ###

This layer converts RGB-coded images to L`*`a`*`b`*`-coded images.

Specifically, it applies the two transformations described on these Wikipedia pages:
  1. http://en.wikipedia.org/wiki/CIE_1931_color_space#Construction_of_the_CIE_XYZ_color_space_from_the_Wright.E2.80.93Guild_data
  1. http://en.wikipedia.org/wiki/Lab_color_space#The_forward_transformation

I'll reproduce the transformation here. Assuming the input RGB values  are in the range 0-1:

![http://cuda-convnet.googlecode.com/svn/wiki/images/rgb_to_xyz.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/rgb_to_xyz.gif)

![http://cuda-convnet.googlecode.com/svn/wiki/images/xyz_to_lab.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/xyz_to_lab.gif)

where

![http://cuda-convnet.googlecode.com/svn/wiki/images/f_lab.png](http://cuda-convnet.googlecode.com/svn/wiki/images/f_lab.png)

**Note that an underlying assumption in this layer is that its input is in the range 0-1. This assumption is violated by the default CIFAR-10 data provider, which produces zero-mean data.**

This layer is specified like this:

```
[lab]
type=rgb2lab
inputs=data
center=true
```

When **center** is true, 50 will be subtracted from the final value of L`*`. Otherwise, L`*` will be left as given by the formulas above. (Note that the range of L`*` in the above formulas is 0-100, while the ranges of a`*` and b`*` are centered at 0).

# Layer parameter file - basic features #

In this file you can specify learning parameters that you may want to change during the course of training (learning rates, etc.). You can find examples of this kind of file [here](https://code.google.com/p/cuda-convnet2/source/browse/#git%2Flayers).

The idea is that you'll use one file to define the net's architecture, which stays fixed for the duration of training, and another file to define learning hyper-parameters, which you can change while the net is training.

I'll now go through the learning hyper-parameters that the above-defined layers take. With a few exceptions, only layers with weights require learning hyper-parameters.

## Weight layer parameters ##

Here's how to specify learning hyper-parameters for layers with weights (convolutional layers, locally-connected layers, and fully-connected layers).

```
[conv32]
epsW=0.001
epsB=0.002
momW=0.9
momB=0.9
wc=0
```

This specifies the learning hyper-parameters for the layer **conv32**. Here's what the parameters mean:
| **Parameter** | **Meaning** |
|:--------------|:------------|
|epsW=0.001 |the weight learning rate. |
|epsB=0.002 |the bias learning rate. |
|momW=0.9|the weight momentum. |
|momB=0.9|the bias momentum. |
|wc=0 | the L2 weight decay (applied to the weights but not the biases).|

Given these values, the update rule for the weights is:
```
weight_inc[i] := momW * weight_inc[i-1] - wc * epsW * weights[i-1] + epsW * weight_grads[i]
weights[i] := weights[i-1] + weight_inc[i]
```

where `weight_grads[i]` is the average gradient over minibatch `i`. The update rule for biases is the same except that there is no bias weight decay.

If you want to not learn the weights in a particular layer (perhaps because you've [initialized them in some awesome way](LayerParams#Initializing_weights_from_an_outside_source.md)), you can set its `epsW` and `epsB` parameters to 0, and then that layer won't be learned. The net will run faster as a result.

Additionally, you can specify learning rate **schedules** as opposed to constants. For example, if you want the learning rate to start at 0.01 and decay exponentially during training such that it finishes at 0.0001, you can write

```
epsW=exp[base=0.01;tgtFactor=100]
```

There are a few more learning rate schedules, and they're described on the LearningRates page.

## Cost layer parameters ##

Cost layers, such as the **logprob** and **sqdiff** layers defined above, take one parameter:
```
[logprob]
coeff=1
```

... a scalar coefficient of the objective function. This provides an easy way to tweak the "global" learning rate of the network. Note, however, that tweaking this parameter is not equivalent to tweaking the **epsW** parameter of every layer in the net if you use weight decay. That's because tweaking **coeff** will leave the effective weight decay coefficient (`epsW*wc`) unchanged. But tweaking **epsW** will of course cause a change in `epsW*wc`.

## Local response/contrast normalization layer parameters ##
The **rnorm**, **cnorm**, and **cmrnorm** layer types two parameters:

```
scale=0.0000125
pow=0.75
```

These are the _alpha_ and _beta_, respectively, in the equations in [LayerParams#Local\_response\_normalization\_layer\_(same\_map)](LayerParams#Local_response_normalization_layer_(same_map).md) and the two section following it.

## Dropout layer parameters ##

[Dropout layers](https://code.google.com/p/cuda-convnet2/wiki/LayerParams#Dropout_layer) take one parameter: a keep probability.

```
[dropout1]
keep=0.75
```

This indicates that neurons should be dropped out independently with probability 0.25.

# Layer definition file  - slightly more advanced features #
## Block sparse convolution layer ##

This is a convolutional layer in which each filter only looks at _some_ of the input channels.

```
[conv32-2]
type=conv
inputs=cnorm1
groups=4
channels=32
filters=32
padding=2
stride=1
filterSize=5
neuron=relu
initW=0.0001
partialSum=1
sharedBiases=false
```

The primary novelty here is the **groups=4** parameter. Together with the **filters=32** parameter, they state that this convolutional layer is to have 4 groups of 32 filters. Each filter will connect to (i.e. sum over) 8 input channels (a quarter of the total number of input channels).

The following diagram depicts the kind of operation that this layer will perform (note however that the diagram depicts two filters per group rather than 32).

![http://cuda-convnet.googlecode.com/svn/wiki/images/conv-diagram.png](http://cuda-convnet.googlecode.com/svn/wiki/images/conv-diagram.png)

**Block sparsity may be used in convolutional layers as well as in locally-connected, unshared layers. Simply replace type=conv with type=local to do this in locally-connected, unshared layers.**

### Performance notes ###
  * Block sparse convolutional layers are just as efficient computationally as non-sparse convolutional layers.
  * The computation will be most efficient when `channels / groups` is divisible by 16 and `filters` is divisible by 32.

## Layers with multiple inputs ##

All of the layers with weights (convolutional, locally-connected, and fully-connected) can take multiple layers as input. When there are multiple input layers, a separate set of weights (filters, if you like) is used for each input. Each input/weight pair produces some output, and the final output of the layer is the summation of all those outputs. This implies that all input/weight pairs must produce output of equivalent dimensions.

For example, you can write a convolutional layer with two input layers like this:
```
[conv5b]
type=conv
inputs=conv4b,conv5
filters=64,64
padding=2,2
stride=1,1
filterSize=5,5
channels=64,64
initW=0.01,0.01
initB=1
partialSum=13
groups=1,1
sharedBiases=true
```

This convolutional layer takes the two layers **conv4b** and **conv5** as input. Since there are two sets of weights in this layer, you must specify two values for most of the parameters which define the operation that those weights perform on their respective inputs. Some of the parameters remain unary, for example the **initB** parameter, since there is still only one bias vector for all the outputs.

As mentioned above, you must choose sets of parameters that lead to convolutions which produce equivalently-sized output vectors.

A locally-connected, unshared layer with multiple inputs can be specified in exactly the same way, replacing **type=conv** with **type=local**.

A fully-connected layer with multiple inputs can be specified like this:

```
[fc1000]
type=fc
outputs=1000
inputs=pool3,probs
initW=0.001,0.1
```

In a fully-connected layer, you don't have to do anything to make the number of output values in the two computations equivalent. The number of output values is defined by the **outputs** parameter.

## Weight sharing between layers ##

Layers may share their weights with other layers of the same kind. One of the things this allows you to do is to write recurrent networks. For example, a layer may take as input the output of another layer which uses **the same weights** -- so in effect the layer is taking its own output (in a previous "time step") as input.

All layers with weights (convolutional, locally-connected, and fully-connected) can share weights. To specify that a layer should use the weight matrix from another layer, add the **weightSource** parameter to the layer's definition file.

Here's an example:

```
[conv7]
type=conv
inputs=conv5
...
weightSource=conv6
```

This says that the layer **conv7** will use the same weight matrix as **conv6**. Of course this requires that the weight matrix of **conv6** have the same shape as the weight matrix that **conv7** would have had without sharing.

Note also that the momentum and weight decay parameters given for **conv7** in the layer parameter file will be ignored. Only layers which own their weight matrix apply momentum and weight decay. Additionally, the **initW** parameter will also be ignored, for obvious reasons.

Now suppose **conv6** has multiple weight matrices (as it would if it took multiple inputs). If you want **conv7** to share its weight matrix with the **third** weight matrix of **conv6**, you can do it like this:

```
[conv7]
type=conv
inputs=conv5
...
weightSource=conv6[2]
```

The bracketed index after the layer name selects a weight matrix from **conv6**. Indices start at 0. If an index is not given, 0 is implied.

Now suppose **conv7** itself takes three inputs. If you want **conv7** to take its first and third weight matrix from other layers, you can do it like this:

```
[conv7]
type=conv
inputs=conv6,conv5,conv4
...
weightSource=conv3,,conv5[1]
```

Hopefully by now it is clear what this means. The first weight matrix will be taken from **conv3`[0]`**, the second weight matrix will be owned by **conv7** -- not taken from anywhere -- and the third weight matrix will be taken from **conv5`[1]`**.

One final example. Suppose you want **conv7** to take three inputs but apply the same (newly-defined) filters to all three. In other words, you want **conv7** to share its weights with itself, amongst its three inputs. Then you can write:

```
[conv7]
type=conv
inputs=conv6,conv5,conv4
...
weightSource=,conv7,conv7
```

This says that the first weight matrix of **conv7** will be new, but the remaining two will be equivalent to the first.

Note that in all cases, biases are never shared between layers. Each layer has its own bias vector.

## Initializing weights from an outside source ##

By default, the net initializes the weights randomly from a normal distribution with standard deviation given by the `initW` parameter. You may want to initialize weights from a different distribution, or perhaps from another model, or from wherever. To do this, add the `initWFunc` parameter to your weight layer, like so:

```
[conv8]
type=conv
inputs=conv6,conv5
...
initWFunc=winitfile.makew(0.5,3)
```

This says that the weight matrices in this layer will come from a function named `makew` in a python file named `winitfile.py`, located in the same directory as the main code. So now you have to write a python function which will return the weight matrices for this layer.

Here is a valid `winitfile.py` file for this case:
```
import numpy as n
import numpy.random as nr

def makew(name, idx, shape, params=None):
    stdev, mean = float(params[0]), float(params[1])
    rows, cols = shape
    return n.array(mean + stdev * nr.randn(rows, cols), dtype=n.single)
```

This doesn't do anything useful, really -- it just initializes the weights from a normal distribution with mean 3 and standard deviation 0.5.

But the above example should make clear the constraints on the weight-creating function.
  * It must take four parameters:
    1. `name`: the layer name (in this example it would be **conv8**).
    1. `idx`: the weight matrix index within that layer. In this example it would be 0 when called to initialize the **conv6**-connected weight matrix and 1 when called to initialize the **conv5**-connected weight matrix.
    1. `shape`: a 2-tuple of (rows, columns) indicating the dimensions of the required weight matrix.
    1. `params`: a keyword parameter which receives a list of string parameters that you can specify in the `initWFunc` definition (as we did above).
  * It must return a `numpy.ndarray` object consisting of single-precision floats.

A typical use case for this might be to have the `makew` function load a weight matrix from some other model and return it. Note that you can easily load weight matrices stored as Matlab objects with the [scipy.io.loadmat](http://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) function.

Note that you don't have to pass parameters to your weight-creating function. The following line is also valid:
```
initWFunc=winitfile.makew
```

In this case the `params` argument in `makew` will be the empty list `[]`.

You might be wondering what dimensions (shapes) the weight matrices that your function returns should have. The following table lists this:

| **Layer type** | **Matrix rows** | **Matrix columns** | **Notes** |
|:---------------|:----------------|:-------------------|:----------|
|[Fully-connected](LayerParams#Fully-connected_layer.md)|`input size`|`output size`|  |
|[Convolutional](LayerParams#Convolution_layer.md)|`filterSize`<sup>2</sup> x `filterChannels`|`filters`| `filterChannels` is the number of channels each filter is connected to. This is equal to the number of channels in the input unless you're using [sparsity](LayerParams#Block_sparse_convolution_layer.md).|
|[Locally-connected](LayerParams#Locally-connected_layer_with_unshared_weights.md)|`outputsX`<sup>2</sup> x `filterSize`<sup>2</sup> x `filterChannels`|`filters`| This layer produces output of shape `filters` x `outputsX` x `outputsX`. So `outputsX` is the number of filters in the _x_ (equivalently, _y_) direction in this layer.|

At present, the function you define **must** return a weight matrix for every input that your layer takes. This means that it's impossible to define a layer in which one weight matrix is shared from another layer while another weight matrix is returned by a custom initialization file. In the future this should be possible.

### Initializing biases ###

Each weight layer has a learned bias vector, and it too can be initialized from an outside source. To do this, add the `initBFunc` parameter to your layer definition:

```
[conv8]
type=conv
inputs=conv6,conv5
...
initWFunc=winitfile.makew(0.5,3)
initBFunc=winitfile.makeb
```

The bias vector will now be initialized from the function `makeb` in the python file `winitfile.py`. Here's a valid `makeb` function:

```
...

def makeb(name, shape, params=None):
    return n.array(0.01 * nr.randn(shape[0], shape[1]), dtype=n.single)
```

The only difference between the interfaces for the `makew` function and the `makeb` function is that the `makeb` function takes no `idx` parameter since each layer only has one bias vector.

The table below lists the shapes of the bias vectors for the various weight layers:

| **Layer type** | **Matrix rows** | **Matrix columns** | **Notes** |
|:---------------|:----------------|:-------------------|:----------|
|[Fully-connected](LayerParams#Fully-connected_layer.md)|`1`|`output size`|  |
|[Convolutional](LayerParams#Convolution_layer.md) - shared biases|`1`|`filters`|  |
|[Convolutional](LayerParams#Convolution_layer.md) - non-shared biases|`1`|`filters` x `outputsX`<sup>2</sup>| This layer produces output of shape `filters` x `outputsX` x `outputsX`. So `outputsX` is the number of filter applications in the _x_ (equivalently, _y_) direction in this layer. |
|[Locally-connected](LayerParams#Locally-connected_layer_with_unshared_weights.md)|`1`|`filters` x `outputsX`<sup>2</sup>| This layer produces output of shape `filters` x `outputsX` x `outputsX`. So `outputsX` is the number of filters in the _x_ (equivalently, _y_)direction in this layer.|

# Layer parameter file  - slightly more advanced features #
## Weight layers with multiple inputs ##
Layers with weights (convolutional, locally-connected, and fully-connected) which also take multiple inputs must specify a learning rate, weight momentum, and weight decay coefficient for each of their weight matrices.

For example, the parameter file definition for the layer **conv5b** (defined above) is this:

```
[conv5b]
epsW=0.001,0.001
epsB=2
momW=0.9,0.9
momB=0.9
wc=0.001,0.001
```

# Training the net #

See TrainingNet for details about how to actually cause training to happen.