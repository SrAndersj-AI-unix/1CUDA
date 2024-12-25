# Archivo: usarModelo.py
import torch
import torch.nn as nn

# Cargar el modelo
def cargar_modelo(ruta_modelo):
    model = nn.Linear(1, 1).to('cuda')  # Asegúrate de que la arquitectura coincide
    model.load_state_dict(torch.load(ruta_modelo))  # Cargar los pesos
    model.eval()  # Establecer el modelo en modo de evaluación
    return model

# Pedir entrada al usuario
def predecir():
    modelo = cargar_modelo("modelo_entrenado.pth")  # Ruta al modelo entrenado
    while True:
        try:
            entrada = float(input("Ingresa un valor para x: "))  # Entrada del usuario
            x_input = torch.tensor([[entrada]], device='cuda')  # Crear tensor de entrada
            y_pred = modelo(x_input)  # Predicción
            print(f"Predicción para x={entrada}: {y_pred.item()}")
        except ValueError:
            print("Por favor, ingresa un número válido.")
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break

# Llamar a la función principal
if __name__ == "__main__":
    predecir()
