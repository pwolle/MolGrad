import tensorflow as tf

from adabelief_tf import AdaBeliefOptimizer
from model.transformer import Transformer
from loss import molecule_sdiffusion_loss
from data import get_gdbs
from tqdm import tqdm


num_atoms = 6

num_layers = 12
bond_depth = 256
atom_depth = 512
num_heads = 16

name = f'model_{num_atoms}_{num_layers}_{bond_depth}_{atom_depth}_{num_heads}'


dataset = get_gdbs(128, num_atoms)

model = Transformer(4, 64, 256, 4)

b, a = next(iter(dataset))
model(b, a)  # to init model


opt = AdaBeliefOptimizer(
    1e-4,
    min_lr=5e-6,
    epsilon=1e-16,
    rectify=True,
    total_steps=20000,
    warmup_proportion=0.05,  # ~ ?
    print_change_log=False)


@tf.function
def train_step(bonds, atoms):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)

        loss = molecule_sdiffusion_loss(bonds, atoms, model)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


step = 0
writer = tf.summary.create_file_writer(f'logs/gdb/' + name)

with writer.as_default():

    for epoch in tqdm(range(20000 // 7)):

        for bonds, atoms in dataset:
            loss = train_step(bonds, atoms)

            tf.summary.scalar('loss', loss, step=step)
            writer.flush()

            step += 1


model.save_weights('model/saved/gdb/' + name)
