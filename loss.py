import tensorflow as tf


def invsigmoid(x, eps=0.):
    x = x * (1 - eps)
    x = x + (eps / 2)
    return tf.math.log(x / (1 - x))


def reduce_fsbm(x):
    return (1 / x.shape[0]) * tf.reduce_sum(x)


def broadcast(x, y):
    if type(x) == float or type(x) == int:
        return x
    x_shape = list(x.shape) + [1] * (len(y.shape) - len(x.shape))
    x = tf.reshape(x, x_shape)
    return x


def mask_diagonal(x):
    m1 = tf.ones_like(x)
    me = tf.eye(x.shape[1], batch_shape=[x.shape[0]])
    m = m1 - me[..., tf.newaxis]
    return x * m


def preprocess_bond_noise(z):
    zT = tf.transpose(z, [0, 2, 1, 3])
    z = (z + zT) * (0.5 * 1.41421)
    z = mask_diagonal(z)
    return z


def interpolate(x, y, t):
    return x + broadcast(t, x) * (y - x)


def s_diffusion(x, t, z_projection=lambda x: x):
    z = tf.random.normal(x.shape)
    z = z_projection(z)

    zs = tf.nn.sigmoid(z)
    x_t = interpolate(x, zs, t)

    return x_t, z


@tf.function
def smol_score_l1(atoms, bonds, model):
    """
    l1 loss on molecular graphs for a s-diffusion score model
    also returns dictionaries for the summary writer

    """

    t = tf.random.uniform([atoms.shape[0]], 0, 1)

    atoms_t, atoms_z = s_diffusion(atoms, t)
    bonds_t, bonds_z = s_diffusion(bonds, t, preprocess_bond_noise)

    atoms_h, bonds_h = model(atoms_t, bonds_t)

    atoms_j = reduce_fsbm(tf.math.abs(atoms_h + atoms_z))
    bonds_j = reduce_fsbm(tf.math.abs(bonds_h + bonds_z))

    j = atoms_j + bonds_j

    scalar_dir = {
        'j': j,
        'j_atoms': atoms_j,
        'j_bonds': bonds_j}

    hist_dir = {
        'aotms_h': atoms_h,
        'atoms_z': atoms_z,
        'bonds_h': bonds_h,
        'bonds_z': bonds_z, }

    return j, scalar_dir, hist_dir


@tf.function
def smol_regression_l1(atoms, bonds, y, model):
    """
    l1 loss on molecular graphs for a simple regression model
    also returns dictionaries for the summary writer

    """

    t = tf.random.uniform([atoms.shape[0]], 0, 1)

    atoms_t, _ = s_diffusion(atoms, t)
    bonds_t, _ = s_diffusion(bonds, t, preprocess_bond_noise)

    atoms_t = invsigmoid(atoms_t, eps=1e-2)
    bonds_t = invsigmoid(bonds_t, eps=1e-2)

    h = model(atoms_t, bonds_t)
    j = reduce_fsbm(tf.math.abs(h - y))

    j_logp = reduce_fsbm(tf.math.abs(h[:, 0] - y[:, 0]))
    j_qed = reduce_fsbm(tf.math.abs(h[:, 1] - y[:, 1]))
    j_sas = reduce_fsbm(tf.math.abs(h[:, 2] - y[:, 2]))

    scalar_dir = {
        'j': j,
        'j_logp': j_logp,
        'j_qed': j_qed,
        'j_sas': j_sas}

    hist_dir = {
        'h': h,
        'h_logp': h[:, 0],
        'h_qed': h[:, 1],
        'h_sas': h[:, 2]}

    return j, scalar_dir, hist_dir


def write_summary(scalars, hists, step, writer, do_hists=False):
    for k, v in scalars.items():
        tf.summary.scalar(k, v, step=step)

    if do_hists:
        for k, v in hists.items():
            tf.summary.histogram(k, v, step=step)

            s = tf.math.reduce_std(v)
            tf.summary.scalar(k + '_stddev', s, step=step)

    writer.flush()


if __name__ == '__main__':
    # atoms = tf.random.normal([32, 6, 5])
    # bonds = tf.random.normal([32, 6, 6, 3])

    # model = lambda a, b: (a, b)

    # smol_score_l1(atoms, bonds, model)

    print(invsigmoid(1., 1e-2))
