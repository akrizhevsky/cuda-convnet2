# Data #

The [example networks](https://code.google.com/p/cuda-convnet2/source/browse/#git%2Flayers) are meant to be trained on the ILSVRC 2012 dataset, which you can download [here](http://www.image-net.org/download-images). Note that you'll have to create an account on www.image-net.org to do this.

You'll need to download the training images for task 1 & 2, the validation images for all tasks, and the development kit for task 1 & 2.

Once you've downloaded the data, you should have a directory with the following three files:

```
ILSVRC2012_img_train.tar
ILSVRC2012_img_val.tar
ILSVRC2012_devkit_t12.tar.gz
```

This project comes with a script to generate training batches from these three files. To run it, first [compile](Compiling.md) `cuda-convnet2`, and then, from the main project directory, type

```
cd make-data
python make-data.py --src-dir /usr/local/storage/akrizhevsky/ilsvrc-2012-tars --tgt-dir /usr/local/storage/akrizhevsky/ilsvrc-2012-batches
```

substituting your own source and target directories. The target directory will be created if it does not exist. **You'll need about 40GB of disk space in the target directory using the default 256x256 target image size.**

You should see output that looks like this (look at the comments in [make-data.py](https://code.google.com/p/cuda-convnet2/source/browse/make-data/make-data.py) for the meanings of the variables printed below):

```
CROP_TO_SQUARE: True
OUTPUT_IMAGE_SIZE: 256
NUM_WORKER_THREADS: 8
Loaded synset tars.
Building training set image list (this can take 10-20 minutes)...
0% ... 10% ... 20% ... 30% ... 40% ... 50% ... 60% ... 70% ... 80% ... 90% ... done
Writing training batches...
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_0 (training batch 1 of 418) (14.28 sec)
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_1 (training batch 2 of 418) (13.74 sec)
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_2 (training batch 3 of 418) (13.72 sec)

...

Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_415 (training batch 416 of 418) (14.34 sec)
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_416 (training batch 417 of 418) (14.40 sec)
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_417 (training batch 418 of 418) (0.60 sec)
Writing validation batches...
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_1000 (validation batch 1 of 17) (5.09 sec)
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_1001 (validation batch 2 of 17) (3.83 sec)
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_1002 (validation batch 3 of 17) (4.44 sec)

...

Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_1014 (validation batch 15 of 17) (4.22 sec)
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_1015 (validation batch 16 of 17) (4.62 sec)
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/data_batch_1016 (validation batch 17 of 17) (1.02 sec)
Wrote /usr/local/storage/akrizhevsky/ilsvrc-2012-batches/batches.meta
All done! ILSVRC 2012 batches are in /usr/local/storage/akrizhevsky/ilsvrc-2012-batches
```

which indicates that the script has generated training and validation batches at the target directory. The training batches are `data_batch_0` ... `data_batch_417` and the validation batches are `data_batch_1000` ... `data_batch_1016`. On my machine this process takes about two hours.

Once the data is generated, you can try to [train an example network](TrainingExample.md).