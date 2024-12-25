# Archivo: 3.2pruebaModeloSimplePytorch.py
import torch
import torch.nn as nn
import torch.optim as optim

# Datos ficticios
x = torch.randn(100, 1).to('cuda')  # Entradas en GPU
y = 3 * x + 2 + torch.randn(100, 1).to('cuda')  # Salida con ruido

# Modelo lineal simple
model = nn.Linear(1, 1).to('cuda')

# Función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entrenamiento
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item()}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "modelo_entrenado.pth")
print("Modelo entrenado guardado como 'modelo_entrenado.pth'")
