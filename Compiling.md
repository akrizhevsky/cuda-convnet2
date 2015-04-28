<font color='red'><b>Note</b>:</font> A Kepler-generation GPU with shader model capability 3.5 or greater is required to run this code. This includes the chips GK110 and GK114, which can be found on the GPUs Tesla K20, Tesla K20x, Tesla K40, GeForce Titan, and GeForce GTX 780, among others. Older GPUs, including GK104-based GPUs such as the Tesla K10 and GeForce 680, won't work.

I've tested this code on modern 64-bit Ubuntu systems (12.04 and newer).

## Required libraries ##

Here's what I hope to be a complete list of prerequisite libraries. Please [email me](mailto:akrizhevsky@gmail.com) if I'm wrong.

| **Library** | **Ubuntu package name** |
|:------------|:------------------------|
| Python development libraries/headers | python-dev |
| Numpy | python-numpy |
| Scipy | python-scipy |
| Python libmagic bindings | python-magic |
| Matplotlib | python-matplotlib |
| ATLAS development libraries/headers | libatlas-base-dev |
| JPEG decompression | libjpeg-dev |
| OpenCV | libopencv-dev |
| git source control system | git |

You can install all of these with the command

```
sudo apt-get install python-dev python-numpy python-scipy python-magic python-matplotlib libatlas-base-dev libjpeg-dev libopencv-dev git
```

And of course you also need to install the CUDA toolkit and CUDA SDK. I have tested this code with CUDA 5.5 and CUDA 6.0.

The configuration that works the best for me is CUDA 6.0 and driver version 340.24.

## Checking out the code ##

You can check out the code with this line:

```
git clone https://code.google.com/p/cuda-convnet2/
```

## Compiling ##

In the main directory, you'll find the [build.sh](https://code.google.com/p/cuda-convnet2/source/browse/build.sh) file. Fill in the environment variables in that file (the included defaults are the proper values on _my_ system, but almost certainly not on yours).

Then, type
```
sh build.sh
```

If all goes well, in a few minutes the compilation will finish successfully. [Email me](mailto:akrizhevsky@gmail.com) in case of trouble.