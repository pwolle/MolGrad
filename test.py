def pad(a, b, *g):
    r = (a, b) + g
    return r


print(pad(1, 2, 3))