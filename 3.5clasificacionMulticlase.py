import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset



# Datos simulados
X = torch.rand(150, 4)  # 150 muestras, 4 características
y = torch.randint(0, 3, (150,))  # Clases: 0, 1, 2

# Dividimos en entrenamiento y prueba
X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]


print(X_train.shape, y_train.shape)
print(X, y)