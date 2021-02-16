import tensorflow as tf


def flat_sum_batch_mean(x):
    return tf.reduce_sum(x) / x.shape[0]


def broadcast(x, y):
    if type(x) == float or type(x) == int:
        return x
    x_shape = list(x.shape) + [1] * (len(y.shape) - len(x.shape))
    x = tf.reshape(x, x_shape)
    return x


def interpolate(x, y, t):
    return x + t * (y - x)


def preprocess_bond_noise(z):
    zT = tf.transpose(z, [0, 2, 1, 3])
    z = (z + zT) * (0.5 * 1.41421)
    z = mask_diagonal(z)
    return z


def mask_diagonal(x):
    m1 = tf.ones_like(x)
    me = tf.eye(x.shape[1], batch_shape=[x.shape[0]])[..., tf.newaxis]
    m = m1 - me
    return x * m


if __name__ == "__main__":
    a = tf.ones([1, 4, 4, 3])  # + 0.5

    a = mask_diagonal(a)[0]

    a = tf.transpose(a, [2, 0, 1])
    print(a)
