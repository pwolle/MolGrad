import tensorflow as tf

from adabelief_tf import AdaBeliefOptimizer
from model.transformer import RegressionTransformer
from loss import molecule_sdiffusion_regression
from data import get_gdbscrippen
from tqdm import tqdm


dataset = get_gdbscrippen(128, 6)
dataset = dataset.shuffle(1024)

student = RegressionTransformer(4, 64, 128, 4)

b, a, _ = next(iter(dataset))
student(b, a)  # to init model


# opt = tf.keras.optimizers.Adam(1e-4)
opt = AdaBeliefOptimizer(
    1e-4,
    min_lr=5e-6,
    epsilon=1e-16,
    rectify=True,
    total_steps=5000,
    warmup_proportion=0.05,  # ~ ?
    print_change_log=False)


@tf.function
def train_step(bonds, atoms, solubility):
    with tf.GradientTape() as tape:
        tape.watch(student.trainable_variables)

        loss = molecule_sdiffusion_regression(bonds, atoms, solubility, student)

    grads = tape.gradient(loss, student.trainable_variables)
    opt.apply_gradients(zip(grads, student.trainable_variables))
    return loss


step = 0
writer = tf.summary.create_file_writer(f'logs/solv/student6crippenR10')

with writer.as_default():

    for epoch in tqdm(range(5000 // 7)):

        for bonds, atoms, solubility in dataset:
            loss = train_step(bonds, atoms, solubility)

            tf.summary.scalar('loss', loss, step=step)
            writer.flush()

            step += 1


student.save_weights('model/saved/solv/student6crippen')
