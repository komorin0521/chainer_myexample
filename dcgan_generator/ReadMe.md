# OverView
This is the dcgan generator from trained network.
This script is based on chainer github of 'example/dcgan'

# Environments
1. python : 3.5.2
2. chainer : 1.19.0
3. github revision of chainer(


# How to use
1. You can install chainer and you clone chainer examples from git
    see website of chainer

    ```bash
    git clone https://github.com/chainer/chainer
    git checkout a2dab071b95aed899e1602d2049251809837282e
    ```

    I think parhaps the latest version can work well.

2. Training of your dataset
    In chainer example, you can defined dataset using '-i' option

3. Using this script, you generated the images.
   `python generator.py -g ./result/(path_to_generator(.npz)) -o ./path_to_outputfolder -i 100`
   `-i` is optional, which the number of generated images
