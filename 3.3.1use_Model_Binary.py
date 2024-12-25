import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
modelo_cargado = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
modelo_cargado.load_state_dict(torch.load("modelo_clasificacion_binaria.pth"))
modelo_cargado.eval()  # Establecer el modelo en modo de evaluación

# Función para predecir valores
def predecir_y_graficar():
    # Preguntar al usuario si quiere ingresar una lista o un único punto
    tipo_entrada = input("¿Quieres ingresar una lista de puntos (s/n)? ").strip().lower()

    if tipo_entrada in ["s", "sí", "si", "yes"]:  # Lista de puntos
        num_puntos = int(input("¿Cuántos puntos deseas ingresar? "))
        puntos = []
        for i in range(num_puntos):
            print(f"Introduce el punto {i + 1}:")
            x1 = float(input("x1: "))
            x2 = float(input("x2: "))
            puntos.append([x1, x2])
        puntos = torch.tensor(puntos, dtype=torch.float32)
    else:  # Único punto
        print("Introduce un único punto para predecir:")
        x1 = float(input("x1: "))
        x2 = float(input("x2: "))
        puntos = torch.tensor([[x1, x2]], dtype=torch.float32)

    # Realizar las predicciones
    with torch.no_grad():
        predicciones = modelo_cargado(puntos)
        clases = (predicciones > 0.5).float()  # Clasificar en 0 o 1

    # Mostrar las predicciones
    for i, punto in enumerate(puntos):
        print(f"Punto: {punto.numpy()}, Probabilidad: {predicciones[i].item():.4f}, Clase: {int(clases[i].item())}")

    # Graficar los puntos predichos
    generar_grafica(puntos.numpy(), clases.numpy())

# Función para generar la gráfica
def generar_grafica(puntos, clases_predichas):
    # Generar datos ficticios para visualización
    torch.manual_seed(0)
    x = torch.randn(100, 2)
    y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)

    # Mover datos a la CPU para visualización
    x_cpu = x.cpu().numpy()
    y_cpu = y.cpu().numpy()

    # Crear la gráfica
    plt.figure(figsize=(8, 6))

    # Dibujar los datos reales (Clase 0: Azul, Clase 1: Rojo)
    plt.scatter(x_cpu[y_cpu.squeeze() == 0, 0], x_cpu[y_cpu.squeeze() == 0, 1], 
                color="blue", alpha=0.5, label="Clase 0 (Datos reales)")
    plt.scatter(x_cpu[y_cpu.squeeze() == 1, 0], x_cpu[y_cpu.squeeze() == 1, 1], 
                color="red", alpha=0.5, label="Clase 1 (Datos reales)")

    # Dibujar la frontera de decisión
    x_boundary = torch.linspace(-3, 3, 100)
    y_boundary = -(modelo_cargado[0].weight[0, 0].item() * x_boundary + modelo_cargado[0].bias.item()) / modelo_cargado[0].weight[0, 1].item()
    plt.plot(x_boundary, y_boundary, color="black", label="Frontera de decisión")

    # Dibujar los puntos predichos
    for i, punto in enumerate(puntos):
        plt.scatter(punto[0], punto[1], color="black" if clases_predichas[i] == 1 else "purple", 
                    s=100, label=f"Punto predicho (Clase {int(clases_predichas[i])})" if i == 0 else "")

    # Etiquetas y leyenda
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("Clasificación binaria: Datos reales, Predicciones y Puntos predichos")
    plt.savefig("grafica_puntos_predichos.png")
    print("La gráfica ha sido guardada como 'grafica_puntos_predichos.png'")
    plt.show()

# Llamar a la función para predecir y graficar
predecir_y_graficar()
