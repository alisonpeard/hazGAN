import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc
import tracemalloc
# from memory_profiler import profile


def main():
    inputs = keras.Input(shape=(10,))
    out = layers.Dense(1)(inputs)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")

    train = np.random.rand(1000,10)
    label = np.random.rand(1000)
    train = tf.convert_to_tensor(train, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.float32)

    tracemalloc.start()
    i = 0
    while i <= 100:
        i += 1
        model.train_step((train, label))
        if i % 10 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
            del current, peak
        gc.collect() # slows down a little but helps a little
        # tf.keras.backend.clear_session() # slows down and doesn't help


if __name__ == "__main__":
    main()


"""
Gradually increasing memory usage: 
32/32 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.2932
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.2424
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.2353
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.2086
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.2149
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1959
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1783
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1663
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1748
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1726
Current memory usage is 2.29926MB; Peak was 2.37835MB
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1574
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1499
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1463
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1439
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1462
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1349
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1395
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1317
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1188
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1148
Current memory usage is 2.318147MB; Peak was 2.397237MB
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1145
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1136
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1065
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1121
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1065
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1066
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1014
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1039
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1019
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0918
Current memory usage is 2.33112MB; Peak was 2.409813MB
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0965
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0951
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0943
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1000
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0919
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0946
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0902
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0875
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0897
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0902
Current memory usage is 2.344063MB; Peak was 2.423153MB
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0860
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0828
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0810
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0879
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0854
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0842
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0802
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0850
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0890
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0873
Current memory usage is 2.354536MB; Peak was 2.433626MB
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0819
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0839
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0800
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0821
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0812
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0785
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0829
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0820
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0826
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0778
Current memory usage is 2.359903MB; Peak was 2.439033MB
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0804
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0823
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0831
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0838
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0813
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0821
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0817
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0813
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0822
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0827
Current memory usage is 2.363757MB; Peak was 2.442895MB
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0851
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0775
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0777
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0810
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0769
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0833
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0829
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0779
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0807
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0826
Current memory usage is 2.372517MB; Peak was 2.451647MB
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0819
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0774
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0816
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0792
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0815
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0801
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0792
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0809
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0800
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0763
Current memory usage is 2.376809MB; Peak was 2.455898MB
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0800
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0826
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0811
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0772
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0799
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0805
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0772
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0789
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0789
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0791
Current memory usage is 2.384755MB; Peak was 2.463844MB
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0804
"""