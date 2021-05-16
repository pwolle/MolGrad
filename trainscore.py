import tensorflow as tf

from argparse import ArgumentParser
from tqdm import tqdm

from models.schedule import CosSchedule
from models.gnn import Transformer

from data import FixMoleculeDataset
from loss import smol_score_l1, write_summary


parser = ArgumentParser()
parser.add_argument('--run_name', type=str, default='trained')
parser.add_argument('--total_steps', type=int, default=int(200e3))
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--summary_path', type=str, default='logs/gdb6_score/{model}_{run}')
parser.add_argument('--save_path', type=str, default='models/trained/{model}_{run}/model/')

args = parser.parse_args()

dataset = FixMoleculeDataset().get_split('train')
dataset = dataset.repeat(args.total_steps)  # works for dataset_size > batch_size
dataset = dataset.batch(args.batch_size, True)

model = Transformer(num_layers=args.num_layers)

opt = tf.keras.optimizers.Adam(
    CosSchedule(args.lr, args.total_steps))


@tf.function
def train_step(atoms, bonds):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss_val, scalars, hists = smol_score_l1(atoms, bonds, model)

    grads = tape.gradient(loss_val, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val, scalars, hists


writer = tf.summary.create_file_writer(
    args.summary_path.format(model=model, run=args.run_name))


with tf.device('gpu:0'), writer.as_default():
    for i, (atoms, bonds) in enumerate(tqdm(dataset, total=args.total_steps)):
        loss_val, scalars, hists = train_step(atoms, bonds)

        if i % 100 == 0:
            write_summary(scalars, hists, i, writer)

            if i >= args.total_steps:
                break

model.save_weights(args.save_path.format(model=model, run=args.run_name))
