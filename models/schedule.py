import tensorflow as tf
import math


class CosSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_maximum, total_steps, warmup=0.05, lr_minimum=1e-8):
        super(CosSchedule, self).__init__()
        self.lr_maximum = lr_maximum
        self.warmup_steps = total_steps * warmup
        self.total_steps = total_steps - self.warmup_steps
        self.lr_minimum = lr_minimum

    def __call__(self, step):
        warmup_lr = step / self.warmup_steps * self.lr_maximum

        cos = tf.math.cos((step - self.warmup_steps) / self.total_steps * math.pi)
        cos_lr = (cos + 1.) * self.lr_maximum * 0.5

        return tf.math.minimum(warmup_lr, cos_lr) + self.lr_minimum


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    total_steps = int(1e4)
    schedule = CosSchedule(3e-4, total_steps)

    x = np.linspace(0, total_steps, total_steps)
    lrs = schedule(x)

    plt.plot(x, lrs)
    plt.show()
