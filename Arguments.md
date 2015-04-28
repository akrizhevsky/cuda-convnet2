# Introduction #

If you type
```
python convnet.py
```

You should get the following output, briefly describing the command-line arguments that this code expects:

```
convnet.py usage:
    Option                             Description                                                              Default 
    [--check-grads <0/1>           ] - Check gradients and quit?                                                [0]     
    [--color-noise <float>         ] - Add PCA noise to color channels with given scale                         [0]     
    [--conserve-mem <0/1>          ] - Conserve GPU memory (slower)?                                            [0]     
    [--conv-to-local <string,...>  ] - Convert given conv layers to unshared local                              []      
    [--epochs <int>                ] - Number of epochs                                                         [50000] 
    [--feature-path <string>       ] - Write test data features to this path (to be used with --write-features) []      
    [--force-save <0/1>            ] - Force save before quitting                                               [0]     
    [--inner-size <int>            ] - Cropped DP: crop size (0 = don't crop)                                   [0]     
    [--layer-path <string>         ] - Layer file path prefix                                                   []      
    [--load-file <string>          ] - Load file                                                                []      
    [--logreg-name <string>        ] - Logreg cost layer name (for --test-out)                                  []      
    [--mini <int>                  ] - Minibatch size                                                           [128]   
    [--multiview-test <0/1>        ] - Cropped DP: test on multiple patches?                                    [0]     
    [--scalar-mean <float>         ] - Subtract this scalar from image (-1 = don't)                             [-1]    
    [--test-freq <int>             ] - Testing frequency                                                        [57]    
    [--test-one <0/1>              ] - Test on one batch at a time?                                             [1]     
    [--test-only <0/1>             ] - Test and quit?                                                           [0]     
    [--test-out <string>           ] - Output test case predictions to given path                               []      
    [--unshare-weights <string,...>] - Unshare weight matrices in given layers                                  []      
    [--write-features <string>     ] - Write test data features from given layer                                []      
     --data-path <string>            - Data path                                                                        
     --data-provider <string>        - Data provider                                                                    
     --gpu <int,...>                 - GPU override                                                                     
     --layer-def <string>            - Layer definition file                                                            
     --layer-params <string>         - Layer parameter file                                                             
     --save-file <string>            - Save file override                                                               
     --save-path <string>            - Save path                                                                        
     --test-range <int[-int]>        - Data batch range: testing                                                        
     --train-range <int[-int]>       - Data batch range: training   
```

You can see that most arguments have default values, so you don't have to provide them.

# What they mean #

First the optional arguments:
| **Argument** | **Meaning** |
|:-------------|:------------|
| `--check-grads` | [Check the gradients](CheckingGradients.md) |
| `--color-noise` | Add random noise with this standard deviation to all input pixels. This is done in the way described in [section 4.1 of this paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). |
| `--conserve-mem` | Conserve GPU memory by deleting activity and gradient matrices when they're not needed. This will slow down the net but it will allow you to run larger nets.|
| `--conv-to-local` | Convert the specified layers from convolutional to unshared, locally-connected. |
| `--epochs` | How many epochs to train for. An epoch is one pass through the training data. |
| `--load-file` | Load checkpoint from given path. All other arguments become optional when this is given. |
| `--inner-size` | Randomly crop input images to this size during training. |
| `--mini` | Minibatch size. The minibatch size is the number of training cases over which the gradient is averaged to produce a weight update. **For best performance this value should be a multiple of 128.** In particular, this code is in no way designed for on-line learning (i.e. minibatch size of 1). So while a minibatch size of 1 will work, processing 1 image may in general take almost as much time as processing a minibatch of 128 images. |
| `--multiview-test` | When using a data provider that crops images, this indicates that the test error computation should be averaged over several regions of the image rather than just the center region. Note that this will greatly increase the time it takes to compute the test error. So it's best to leave this off until you're done training, at which point you can compute the averaged test error once using this option in conjunction with the `--test-only` option. |
| `--test-freq` | The test error computation / checkpoint saving frequency, in units of training batches. Every **this many** training batches, the net will compute the test error and save a checkpoint.|
| `--test-one` | If multiple testing batches are given, each test output will be computed on only one of the batches. The net will cycle through them in sequence. |
| `--test-only` | Compute the test error and quit. To be used in combination with the `--load-file` option. |
| `--unshare-weights` | Remove the coupling of weight matrices between layers. See [TrainingNet](TrainingNet#Decouple_weight_matrices_between_layers.md) for details. |
| `--write-features` | Generate features from given layer and write them to disk. The features will be generated from the test data (`--test-range` argument) and written to the path given to `--feature-path`. |

Now the mandatory ones:

| **Argument** | **Meaning** |
|:-------------|:------------|
| `--data-path` | The path where the net's [data provider](Data.md) can find its data. |
| `--data-provider` | The [data provider](Data.md) to use to parse the data. |
| `--gpu` | List of GPUs to run on (default is to run on GPU 0). |
| `--layer-def` | The [layer definition](LayerParams#Layer_definition_file.md) file. |
| `--layer-params` | The [layer parameter](LayerParams#Layer_parameter_file.md) file. |
| `--save-file` | Override for full checkpoint saving path. Not necessary if `--save-path` option is given. |
| `--save-path` | The path underneath which the net will save checkpoints. |
| `--test-range` | The data [batches](TrainingExample.md) that the net should use to compute the test error. |
| `--train-range` | The data [batches](TrainingExample.md) that the net should use for training. |