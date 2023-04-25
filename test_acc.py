import numpy as np
import matplotlib.pyplot as plt

categories = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']
a = np.array([0.9593, 0.7294, 0.9929, 0.9968, 0.6930, 0.9973])
plt.bar(categories, a)
plt.title('Piece-wise Precision')
plt.xlabel('Piece Category')
plt.ylabel('Precision')
plt.show()