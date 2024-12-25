import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generar datos ficticios para clasificación binaria
torch.manual_seed(0)
x = torch.randn(100, 2)  # 100 puntos con 2 características cada uno
y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)  # Clase 1 si suma > 0, clase 0 en caso contrario

# Dividir los datos en conjuntos de entrenamiento y prueba
train_x, test_x = x[:80], x[80:]
train_y, test_y = y[:80], y[80:]

# Modelo lineal para clasificación
model = nn.Sequential(
    nn.Linear(2, 1),  # 2 características de entrada, 1 salida
    nn.Sigmoid()  # Activación para probabilidad
)

# Función de pérdida y optimizador
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Entrenamiento
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(train_x)
    loss = criterion(y_pred, train_y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Evaluación
with torch.no_grad():
    y_pred_test = model(x)  # Predicciones para todos los datos
    y_pred_class = (y_pred_test > 0.5).float()  # Clasificaciones predichas

# Mover datos a CPU para visualización
x_cpu = x.cpu().numpy()
y_cpu = y.cpu().numpy()
y_pred_class_cpu = y_pred_class.cpu().numpy()

# Crear la visualización
plt.figure(figsize=(8, 6))

# Dibujar los datos reales (Clase 0: Azul, Clase 1: Rojo)
plt.scatter(x_cpu[y_cpu.squeeze() == 0, 0], x_cpu[y_cpu.squeeze() == 0, 1], 
            color="blue", alpha=0.5, label="Clase 0 (Datos reales)")
plt.scatter(x_cpu[y_cpu.squeeze() == 1, 0], x_cpu[y_cpu.squeeze() == 1, 1], 
            color="red", alpha=0.5, label="Clase 1 (Datos reales)")

# Dibujar las predicciones en verde
plt.scatter(x_cpu[:, 0], x_cpu[:, 1], facecolors='none', edgecolors="green", 
            alpha=0.8, label="Predicciones del modelo")

# Dibujar la frontera de decisión
x_boundary = torch.linspace(-3, 3, 100)
y_boundary = -(model[0].weight[0, 0].item() * x_boundary + model[0].bias.item()) / model[0].weight[0, 1].item()
plt.plot(x_boundary, y_boundary, color="black", label="Frontera de decisión")

# Etiquetas y leyenda
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Clasificación binaria: Datos reales y Predicciones")
plt.savefig("clasificacion_binaria_predicciones_corregido.png")
print("La gráfica ha sido guardada como 'clasificacion_binaria_predicciones_corregido.png'")


# Guardar el modelo entrenado
torch.save(model.state_dict(), "modelo_clasificacion_binaria.pth")
print("El modelo ha sido guardado como 'modelo_clasificacion_binaria.pth'")

