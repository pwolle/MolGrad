import tensorflow as tf

from argparse import ArgumentParser
from tqdm import tqdm

from models.schedule import CosSchedule
from models.gnn import RegressionTransformer

from data import FixMoleculeDataset
from loss import smol_regression_l1, write_summary


parser = ArgumentParser()
parser.add_argument('--run_name', type=str, default='trained')
parser.add_argument('--total_steps', type=int, default=int(200e3))
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--summary_path', type=str, default='logs/gdb6_props/{model}_{run}')
parser.add_argument('--save_path', type=str, default='models/trained/{model}_{run}/model/')

args = parser.parse_args()

dataset = FixMoleculeDataset(supervised=True).get_split('train')
dataset = dataset.repeat(args.total_steps)  # works for dataset_size > batch_size
dataset = dataset.batch(args.batch_size, True)

model = RegressionTransformer()

opt = tf.keras.optimizers.Adam(
    CosSchedule(1e-3, args.total_steps))


@tf.function
def train_step(atoms, bonds, props):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss_val, scalars, hists = smol_regression_l1(atoms, bonds, props, model)

    grads = tape.gradient(loss_val, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val, scalars, hists


writer = tf.summary.create_file_writer(
    args.summary_path.format(model=model, run=args.run_name))


with tf.device('gpu:0'), writer.as_default():
    for i, (atoms, bonds, props) in enumerate(tqdm(dataset, total=args.total_steps)):
        loss_val, scalars, hists = train_step(atoms, bonds, props)

        if i % 100 == 0:
            write_summary(scalars, hists, i, writer)

            if i >= args.total_steps:
                break

model.save_weights(args.save_path.format(model=model, run=args.run_name))
