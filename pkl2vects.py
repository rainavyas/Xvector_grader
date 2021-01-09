import pickle
import numpy as np
import argparse
import sys
import os


def get_vects(obj, F=30000):
    n = len(obj['plp'][0][0][0][0][0]) # dimension of mfcc vector
    N = len(obj['plp'])

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
        print(F_counter)

    return X, M

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify pkl file')
commandLineParser.add_argument('OUT', type=str, help='Specify output pkl file')
commandLineParser.add_argument('--F', default=30000, type=int, help='Specify maximum number of frames in phone instance')


args = commandLineParser.parse_args()
pkl_file = args.PKL
out_file = args.OUT
F = args.F

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/pkl2vects.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

pkl = pickle.load(open(pkl_file, "rb"))
print("Loaded pkl")

# Get the batched tensors
X, M = get_vects(pkl, F)

# Get the output labels
y = (pkl['score'])

# Save to pickle file
pkl_obj = [X.tolist(), M.tolist(), y]
pickle.dump(pkl_obj, open(out_file, "wb"))
