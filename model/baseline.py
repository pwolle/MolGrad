import tensorflow as tf

tl = tf.keras.layers


def get_mlp(units=[512, 512, 512, 512, 2], act=tl.LeakyReLU(0.2)):
    layers = []

    for u in units[:-1]:
        layers.append(tl.Dense(u, activation=act))

    layers.append(tl.Dense(units[-1]))

    model = tf.keras.Sequential(layers)
    return model


if __name__ == "__main__":
    model = get_mlp()
    model(tf.zeros([1, 2]))
    model.summary()
