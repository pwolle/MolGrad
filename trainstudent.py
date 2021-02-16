import tensorflow as tf

from adabelief_tf import AdaBeliefOptimizer
from model.transformer import RegressionTransformer
from loss import molecule_sdiffusion_regression
from data import get_gdbssolve
from tqdm import tqdm


dataset = get_gdbssolve(128, 6)

student = RegressionTransformer(8, 128, 512, 16, atom_in=2)

b, a, _ = next(iter(dataset))
student(b, a)  # to init model


opt = AdaBeliefOptimizer(
    1e-4,
    min_lr=5e-6,
    epsilon=1e-16,
    rectify=True,
    total_steps=20000,
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
writer = tf.summary.create_file_writer(f'logs/solv/student6diff')

with writer.as_default():

    for epoch in tqdm(range(20000 // 7)):

        for bonds, atoms, solubility in dataset:
            loss = train_step(bonds, atoms, solubility)

            tf.summary.scalar('loss', loss, step=step)
            writer.flush()

            step += 1


student.save_weights('model/saved/solv/student6')
