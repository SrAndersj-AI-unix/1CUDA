import torch

# Verifica si CUDA está disponible
print("¿CUDA disponible?:", torch.cuda.is_available())

# Imprime el nombre del dispositivo CUDA
if torch.cuda.is_available():
    print("Nombre del dispositivo CUDA:", torch.cuda.get_device_name(0))
