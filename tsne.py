from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
embeddings = np.load('/DATA/nfsshare/Amartya/check/ori.npy')
tsne = TSNE(random_state = 0, n_iter = 1000, metric = 'cosine')
embeddings2d = tsne.fit_transform(embeddings)
fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(embeddings2d[:,0], embeddings2d[:,1], cmap='autumn')
plt.savefig("/DATA/nfsshare/Amartya/check/ori1.png") 
plt.show()