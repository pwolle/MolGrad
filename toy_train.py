import tensorflow as tf

from model.baseline import get_mlp
from loss import sdiffusion_loss
from data import get_toy

dataset = get_toy(512)

model = get_mlp()
model(next(iter(dataset)))  # to init model

schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 1500, 0.5)

opt = tf.keras.optimizers.Adam(schedule)


@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)

        loss = sdiffusion_loss(x, model)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss


step = 0
writer = tf.summary.create_file_writer(f'logs/synthetic/model')

with writer.as_default():

    for epoch in range(15000):

        for x in dataset:
            loss = train_step(x)

            tf.summary.scalar('loss', loss, step=step)
            writer.flush()

            step += 1

model.save_weights('model/saved/synthetic/model')
