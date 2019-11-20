import numpy as np


def param_size(net):
    total_size = 0

    mods = net.modules()
    sizes = []

    for i, mod in enumerate(mods):
        param = list(mod.parameters())
        for j in range(len(param)):
            sizes.append(np.array(param[j].size()))

    for i in range(len(sizes)):
        s = sizes[i]
        total_size += np.prod(np.array(s)) * 32 / (1024 ** 2)

    return total_size
