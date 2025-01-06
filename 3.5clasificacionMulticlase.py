import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

# Verificar si hay GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Datos simulados
X = torch.rand(150, 4).to(device)  # 150 muestras, 4 características
y = torch.randint(0, 3, (150,)).to(device)  # Clases: 0, 1, 2

# Dividimos en entrenamiento y prueba
X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]

print(X_train.shape, y_train.shape)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Crear datasets
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class MulticlassModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MulticlassModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Primera capa
        self.relu = nn.ReLU()  # Activación
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Capa de salida

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instanciamos el modelo y lo movemos a GPU
model = MulticlassModel(input_size=4, hidden_size=16, num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()  # Función de pérdida
optimizer = optim.Adam(model.parameters(), lr=0.0000000001)  # Optimizador Adam

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()  # Activar modo entrenamiento
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)  # Paso hacia adelante
        loss = criterion(outputs, y_batch)  # Calcular pérdida

        optimizer.zero_grad()  # Reiniciar gradientes
        loss.backward()  # Paso hacia atrás (gradientes)
        optimizer.step()  # Actualizar pesos

        total_loss += loss.item()

   # Imprimir la pérdida cada 10 épocas
    if (epoch + 1) % 100 == 0 or epoch == 0:  # Imprime también la primera época
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

model.eval()  # Modo evaluación
correct = 0
total = 0

with torch.no_grad():  # No calcular gradientes
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)  # Predicciones
        _, predicted = torch.max(outputs, 1)  # Clase con mayor probabilidad
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = correct / total
print(f"Accuracy on test data: {accuracy * 100:.2f}%")


















#-----------------

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

# Convertir tensores a numpy para graficar
X_train_np = X_train.cpu().numpy()
y_train_np = y_train.cpu().numpy()
X_test_np = X_test.cpu().numpy()
y_test_np = y_test.cpu().numpy()

# Obtener predicciones para los datos de prueba
model.eval()  # Modo evaluación
with torch.no_grad():
    test_outputs = model(X_test)
    _, test_predictions = torch.max(test_outputs, 1)
    test_predictions_np = test_predictions.cpu().numpy()

# Colores consistentes para las clases
classes = np.unique(np.concatenate((y_train_np, y_test_np)))
cmap = plt.colormaps.get_cmap("viridis").resampled(len(classes))
colors = {cls: cmap(i / len(classes)) for i, cls in enumerate(classes)}

# Crear combinaciones de características para graficar
feature_combinations = list(combinations(range(X_train_np.shape[1]), 2))

# Graficar datos de entrenamiento
plt.figure(figsize=(15, 10))
for i, (feat1, feat2) in enumerate(feature_combinations, 1):
    plt.subplot(2, len(feature_combinations) // 2, i)
    for cls in classes:
        idx_train = y_train_np == cls
        plt.scatter(X_train_np[idx_train, feat1], X_train_np[idx_train, feat2], 
                    color=colors[cls], alpha=0.6, label=f"Clase {cls}" if i == 1 else "")
    plt.title(f"Entrenamiento: Características {feat1 + 1} vs {feat2 + 1}")
    plt.xlabel(f"Característica {feat1 + 1}")
    plt.ylabel(f"Característica {feat2 + 1}")
    if i == 1:
        plt.legend()

plt.tight_layout()
plt.savefig("model_training_data.png")

# Graficar predicciones del modelo
plt.figure(figsize=(15, 10))
for i, (feat1, feat2) in enumerate(feature_combinations, 1):
    plt.subplot(2, len(feature_combinations) // 2, i)
    for cls in classes:
        idx_test = test_predictions_np == cls
        plt.scatter(X_test_np[idx_test, feat1], X_test_np[idx_test, feat2], 
                    color=colors[cls], alpha=0.8, marker="x", label=f"Clase {cls}" if i == 1 else "")
    plt.title(f"Predicciones: Características {feat1 + 1} vs {feat2 + 1}")
    plt.xlabel(f"Característica {feat1 + 1}")
    plt.ylabel(f"Característica {feat2 + 1}")
    if i == 1:
        plt.legend()

plt.tight_layout()
plt.savefig("model_only_predictions.png")
