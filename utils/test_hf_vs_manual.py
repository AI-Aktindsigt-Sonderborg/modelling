import numpy as np

logits_manual = np.load('../data/logits_manual.npy')
logits_hf = np.load('../data/logits_hf.npy')
logits_hf_old = np.load('../data/logits_hf_old.npy')


labels_manual = np.load('../data/labels_manual.npy')
labels_hf = np.load('../data/labels_hf.npy')
labels_hf_old = np.load('../data/labels_hf_old.npy')

print()