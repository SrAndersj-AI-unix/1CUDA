import torch

# configuracion del dispositivo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'