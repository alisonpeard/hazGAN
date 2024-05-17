#!/bin/bash -x
which python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" # https://www.tensorflow.org/install/pip