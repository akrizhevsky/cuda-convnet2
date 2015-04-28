# Neuron types #

Most of the [layer types](LayerParams.md) take a `neuron=x` parameter, which defines the output nonlinearity of neurons in the layer.

These are the supported neuron types:

| **Parameter** | **Name** | **Function** |
|:--------------|:---------|:-------------|
| `neuron=logistic`|logistic | ![http://cuda-convnet.googlecode.com/svn/wiki/images/logistic.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/logistic.gif)|
|`neuron=tanh[a,b]` |hyperbolic tangent | ![http://cuda-convnet.googlecode.com/svn/wiki/images/tanh.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/tanh.gif)|
| `neuron=relu`|rectified linear | ![http://cuda-convnet.googlecode.com/svn/wiki/images/relu.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/relu.gif)|
| `neuron=brelu[a]`|bounded rectified linear | ![http://cuda-convnet.googlecode.com/svn/wiki/images/brelu.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/brelu.gif)|
| `neuron=softrelu`|soft rectified linear | ![http://cuda-convnet.googlecode.com/svn/wiki/images/softrelu.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/softrelu.gif)|
| `neuron=abs`|absolute value | ![http://cuda-convnet.googlecode.com/svn/wiki/images/abs.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/abs.gif)|
| `neuron=square`|square | ![http://cuda-convnet.googlecode.com/svn/wiki/images/square.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/square.gif)|
| `neuron=sqrt`|square root | ![http://cuda-convnet.googlecode.com/svn/wiki/images/sqrt.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/sqrt.gif)|
| `neuron=linear[a,b]`|linear | ![http://cuda-convnet.googlecode.com/svn/wiki/images/linear.gif](http://cuda-convnet.googlecode.com/svn/wiki/images/linear.gif)|

_x_, _a_, and _b_ above are all scalars.

Adding new types of neurons is easy. See [neuron.cuh](https://code.google.com/p/cuda-convnet2/source/browse/cudaconvnet/include/neuron.cuh) for reference.