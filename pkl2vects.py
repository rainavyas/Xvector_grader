import pkl
import numpy as np


def get_vects(obj, F=8000):
    n = len(obj['plp'][0][0][0][0][0]) # dimension of mfcc vector

    # Define required tensors
    X = np.zeros((N, F, n))
    M = np.zeros((N, F, n))

    for spk in range(N):
        print("On speaker " + str(spk) + "of "+str(N))
        F_counter = 0
        for utt in range(len(obj['plp'][spk])):
            for w in range(len(obj['plp'][spk][utt])):
                for ph in range(len(obj['plp'][spk][utt][w])):
                    for frame in range(len(obj['plp'][spk][utt][w][ph])):
                        if F_counter >= F:
                            continue
                        x = np.array(obj['plp'][spk][utt][w][ph][frame])
                        X[spk][F_counter] = x
                        M[spk][F_counter] = np.ones(n)
                        F_counter += 1

    return X, M
