import torch
import torch.nn as nn

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir el número de características
n_features = 3  # Cambiar según el modelo

# Ruta del modelo guardado
model_path = "modelo_regresion_Multivariado.pth"

# Cargar el modelo
loaded_model = nn.Linear(n_features, 1).to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()
print("Modelo cargado exitosamente.")

# Función para realizar predicciones
def predict_with_model():
    print(f"Ingrese {n_features} valores, presionando Enter después de cada uno:")

    # Leer valores uno por uno
    data = []
    for i in range(n_features):
        try:
            value = float(input(f"Valor {i+1}: "))
            data.append(value)
        except ValueError:
            print("Error: Ingrese un número válido.")
            return

    # Convertir los datos a tensor
    input_tensor = torch.tensor([data], dtype=torch.float32).to(device)

    # Realizar la predicción
    with torch.no_grad():
        prediction = loaded_model(input_tensor)
        print(f"Predicción: {prediction.item():.4f}")

# Llamar a la función
predict_with_model()

