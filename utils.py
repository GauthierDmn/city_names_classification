import numpy as np
import string

all_letters = string.ascii_letters[:26] + " "
n_letters = len(all_letters)

def line_to_tensor(line):
    tensor = np.zeros((len(line), n_letters)) # tensor --> np
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][letter_index] = 1
    return tensor
