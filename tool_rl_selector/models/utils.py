from __future__ import division

def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                # print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


def conv2d_size_out(size, kernel_size = 5, stride = 2, padding = 0):
    return int((size - (kernel_size) + 2 * padding) // stride  + 1)

def prRed(prt):
    return "\033[91m{}\033[00m" .format(prt)


def prGreen(prt):
    return "\033[92m{}\033[00m" .format(prt)


def prYellow(prt):
    return "\033[93m{}\033[00m" .format(prt)


def prLightPurple(prt):
    return "\033[94m{}\033[00m" .format(prt)


def prPurple(prt):
    return "\033[95m{}\033[00m" .format(prt)


def prCyan(prt):
    return "\033[96m{}\033[00m" .format(prt)


def prLightGray(prt):
    return "\033[97m{}\033[00m" .format(prt)


def prBlack(prt):
    return "\033[98m{}\033[00m" .format(prt)


def prAuto(prt):
    if '[INFO]' in prt:
        return prGreen(prt)
    elif '[WARNING]' in prt:
        return prYellow(prt)
    elif '[ERROR]' in prt:
        return prRed(prt)
    return prt
