# Introduction #

This page describes the learning rate schedules supported by `cuda-convnet2`.

# Schedules #

| **Schedule** | **Example** | **Explanation** |
|:-------------|:------------|:----------------|
| Constant | `epsW=0.01` | The learning rate will be 0.01 throughout training. |
| Exponential | `epsW=exp[base=0.01;tgtFactor=1000]` | The learning rate will start at 0.01 and decay exponentially to 0.00001  by the end of training. |
| Discretized exponential | `epsW=dexp[base=0.01;numSteps=4;tgtFactor=1000]` | The learning rate will start at 0.01 and will be divided by a factor of 1000^(1/3) at 25%, 50%, and 75% training progress, such that the final learning rate is 0.00001. |
| Linear | `epsW=linear[base=0.01;tgtFactor=1000]` | The learning rate will start at 0.01 and will decay linearly to 0.00001 by the end of training. |

See https://code.google.com/p/cuda-convnet2/source/browse/cudaconvnet/src/lr.cu for reference.