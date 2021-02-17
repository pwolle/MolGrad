import matplotlib.pyplot as plt
import tensorflow as tf

from data import get_toy
from model.baseline import get_mlp
from generation import langevin_step
from loss.diffusion import sdiffusion


dataset = get_toy(512)
data = next(iter(dataset))

model = get_mlp()
model(data)  # to init model
model.load_weights('model/saved/synthetic/model')


params = {"ytick.color": "w",
          "xtick.color": "w",
          "axes.labelcolor": "w",
          "axes.edgecolor": "w"}

plt.rcParams.update(params)


r = tf.linspace(0, 1, 20)

x, y = tf.meshgrid(r, r)


c = tf.concat([x[..., tf.newaxis], y[..., tf.newaxis]], -1)
c = tf.reshape(c, [-1, 2])

h = model(c)


plt.quiver(c[:, 0], c[:, 1], h[:, 0], h[:, 1], color='tab:orange')

plt.savefig(
    'figs/gradients.png',
    bbox_inches='tight',
    transparent=True,
    dpi=256)
