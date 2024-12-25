import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
    
    # Predicciones explícitas
    y_pred = model(x)  # Aquí están las predicciones (\hat{y})
    
    # Calcular la pérdida
    loss = criterion(y_pred, y)
    
    # Retropropagación y optimización
    loss.backward()
    optimizer.step()
    
    # Imprimir progreso cada 10 épocas
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item()}")

# Imprime los parámetros aprendidos
print("Peso:", model.weight.data)
print("Bias:", model.bias.data)

# Mover datos y predicciones a la CPU para visualizarlos
x_cpu = x.cpu().numpy()
y_cpu = y.cpu().numpy()
y_pred_cpu = y_pred.detach().cpu().numpy()  # Mover \hat{y} a la CPU y desconectar del grafo computacional

# Crear una visualización de los datos
plt.figure(figsize=(8, 6))
plt.scatter(x_cpu, y_cpu, color="blue", alpha=0.7, label="Datos reales (y)")
plt.scatter(x_cpu, y_pred_cpu, color="red", alpha=0.7, label="Predicciones del modelo (\hat{y})")
plt.title("Datos reales vs. Predicciones del modelo", fontsize=14)
plt.xlabel("x (entrada)", fontsize=12)
plt.ylabel("y / \hat{y} (salida)", fontsize=12)
plt.axline((0, 2), slope=3, color="green", linestyle="--", label="Relación real: y = 3x + 2")
plt.legend()
plt.grid(alpha=0.3)

# Guardar la imagen como PNG
plt.savefig("datos_vs_predicciones.png")  # Nombre del archivo
print("La gráfica ha sido guardada como 'datos_vs_predicciones.png'")
