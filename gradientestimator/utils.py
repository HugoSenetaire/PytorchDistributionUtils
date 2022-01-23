import torch
from itertools import combinations
import numpy as np


def get_all_z(dim):
    """Get all possible combinations of z"""
    output = np.zeros((2**dim, dim))
    number = 0
    for nb_pos in range(dim+1):
        combinations_index = combinations(range(dim), r = nb_pos)
        for combination in combinations_index :
            for index in combination :
                output[number,index]=1
            number+=1
    output = torch.tensor(output, dtype=torch.float)

    return output

