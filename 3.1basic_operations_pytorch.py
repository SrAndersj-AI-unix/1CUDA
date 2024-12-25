import torch

# Crear un tensor en la CPU
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor en CPU:", x)

# Mover el tensor a la GPU
x_gpu = x.to('cuda')
print("Tensor en GPU:", x_gpu)

# Realizar operaciones en la GPU
y_gpu = x_gpu * 2
print("Resultado en GPU:", y_gpu)

# Mover el resultado de vuelta a la CPU
y_cpu = y_gpu.to('cpu')
print("Resultado en CPU:", y_cpu)


