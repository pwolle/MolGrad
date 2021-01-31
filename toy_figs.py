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


x = tf.random.normal(data.shape)


plt.rcParams.update({'font.size': 10})

nplots = 4

fig, axs = plt.subplots(
    1, nplots,
    figsize=(10, 10 * nplots),
    sharex=True, sharey=True,
    gridspec_kw={"wspace": 0.1})

ax_num = 0

alpha_0 = 1.
tau = 3.02
N = 501

for i in range(N):

    if i == 0 or i == 100 or i == 200 or i == 500:
        ax = axs[ax_num]
        ax_num += 1

        xs = tf.nn.sigmoid(x)

        ax.scatter(xs[:, 0], xs[:, 1], 2)

        ax.set_xlabel(f'Schritt {i}')
        ax.set_aspect(1)

        if i > 0:
            ax.tick_params(axis='both', which='both', length=0)

    score = model(tf.nn.sigmoid(x))

    alpha = alpha_0 * tf.exp(-tau * i / N)

    x = langevin_step(x, score, alpha, 0.6, 0.1)

    alpha = alpha * 0.994


fig.savefig(
    'figs/generatediffusion2d.png',
    bbox_inches='tight',
    transparent=False,
    dpi=90)


fig, axs = plt.subplots(
    1, nplots,
    figsize=(10, 10 * nplots),
    sharex=True, sharey=True,
    gridspec_kw={"wspace": 0.1})


for i, ax in enumerate(axs[:-1]):

    t = i / (nplots - 1)

    y, _ = sdiffusion(data, t)

    ax.scatter(y[:, 0], y[:, 1], 2)

    ax.set_xlabel(f't={t:.2f}')
    ax.set_aspect(1)

    if i > 0:
        ax.tick_params(axis='both', which='both', length=0)


ax = axs[-1]

t = tf.random.uniform([data.shape[0]])

y, _ = sdiffusion(data, t)

ax.scatter(y[:, 0], y[:, 1], 2)

ax.set_xlabel('t$∼\mathcal{U}\,(0,1)$')
ax.set_aspect(1)

if i > 0:
    ax.tick_params(axis='both', which='both', length=0)


fig.savefig(
    'figs/forwarddiffusion2d.png',
    bbox_inches='tight',
    transparent=False,
    dpi=90)
