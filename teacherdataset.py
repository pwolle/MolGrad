import tensorflow as tf

from model.transformer import RegressionTransformer
from data import get_freesolv
from data.prepare import GDBsSolve


model = RegressionTransformer(4, 64, 256, 4, atom_in=2)

b, a, _ = next(iter(get_freesolv(1)))
model(b, a)  # to init model

model.load_weights('model/saved/solv/teacher6')

dataset = GDBsSolve(model)
dataset.write()
