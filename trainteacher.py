import tensorflow as tf

from adabelief_tf import AdaBeliefOptimizer
from model.transformer import RegressionTransformer
from loss import regression_l1
from data import get_freesolv
from tqdm import tqdm


dataset = get_freesolv(128)

model = RegressionTransformer(4, 64, 256, 4)

b, a, _ = next(iter(dataset))
model(b, a)  # to init model

opt = AdaBeliefOptimizer(
    1e-4,
    min_lr=5e-6,
    epsilon=1e-16,
    rectify=True,
    total_steps=3000,
    warmup_proportion=0.05,  # ~ ?
    print_change_log=False)


@tf.function
def train_step(bonds, atoms, solubility):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)

        loss = regression_l1(bonds, atoms, solubility, model)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


step = 0
writer = tf.summary.create_file_writer(f'logs/solv/teacher6')


# since the whole dataset fits into one batch
bonds, atoms, solubility = next(iter(dataset))


with writer.as_default():

    for epoch in tqdm(range(3000)):

        loss = train_step(bonds, atoms, solubility)

        tf.summary.scalar('loss', loss, step=step)
        writer.flush()

        step += 1


model.save_weights('model/saved/solv/teacher6')
