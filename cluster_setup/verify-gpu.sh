#!/bin/bash -x
python -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow as tf;print(tf.config.list_physical_devices('CPU'))"
