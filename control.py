import torch


def de_control(x: torch.Tensor):
    # x: torch.Size([20])
    # 22 = 8 + 3 + 8 + 3
    c1, c2, c3, c4 = map(lambda a: a.argmax(), x.split([8, 3, 8, 3]))
    print(f'c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}')
    return c1, c2, c3, c4
