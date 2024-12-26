import torch
import torch.nn as nn
import torch.optim as optim


# Datos simulados (features y etiquetas)
X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [10.0, 11.0, 12.0]], dtype=torch.float32)  # Matriz de características (4 muestras x 3 features)

y = torch.tensor([[14.0],
                  [32.0],
                  [50.0],
                  [68.0]], dtype=torch.float32)  # Etiquetas (4 muestras x 1 salida)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print( device)


#Mover los tensores a la GPU

X= X.to(device)
y=y.to(device)

print("tensor X es:",X)
print("\n")
print("tensor y",y)


# Definir el modelo
n_features = X.shape[1]
model= nn.Linear(n_features,1).to(device)

# Imprimir el modelo y los pesos iniciales
print("model is :",model)
print("--")
print("model weight is :",model.weight)
print("--")
# Definir la función de pérdida y el optimizador

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
print(criterion)
print("--")
print(optimizer)


print("\n")
print("--")
print("--")

# Entrenamiento del modelo
n_epochs = 1000

for epoch in range(n_epochs):

    prediction = model(X)


    loss = criterion(prediction, y)
    # print("loss is:",loss)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')



import matplotlib.pyplot as plt

# Obtener predicciones finales
with torch.no_grad():  # Desactivar gradientes durante la evaluación
    predictions = model(X).cpu().numpy()  # Mover predicciones a la CPU para graficar
    y_true = y.cpu().numpy()  # Mover etiquetas a la CPU para graficar

# Graficar resultados
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_true)), y_true, label="Valores Reales", color="blue")
plt.scatter(range(len(predictions)), predictions, label="Predicciones", color="red", marker='x')
plt.title("Comparación entre Valores Reales y Predicciones")
plt.xlabel("Índice de Muestra")
plt.ylabel("Valores")
plt.legend()
plt.grid(True)

# Guardar la imagen en formato PNG
output_path = "resultados_regresion.png"  # Nombre del archivo de salida
plt.savefig(output_path, format="png", dpi=300)  # Guardar con alta calidad (300 dpi)
print(f"Gráfico guardado como {output_path}")




# Guardar el modelo entrenado
model_path = "modelo_regresion_Multivariado.pth"
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")
