import tensorflow as tf


def langevin_step(x, score, alpha, lmbda=1, diverge=0):
    z = tf.random.normal(x.shape)
    grad_step = alpha * (1 / lmbda) * score
    diffusion_step = 2 * lmbda * tf.math.sqrt(alpha) * z
    divergence_step = diverge * x
    return x + grad_step + diffusion_step + divergence_step


def generate(x, model, N, tau=20, alpha_0=1, lmbda=1):

    for i in range(N):

        alpha = tf.math.exp(-i * tau / N)

        score = model(tf.nn.sigmoid(x))

        x = langevin_step(x, score, alpha)

    return tf.nn.sigmoid(x)
