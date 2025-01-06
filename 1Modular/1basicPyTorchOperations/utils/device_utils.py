from config import DEVICE

def move_to_device(tensor,device=None)
    """ Mueve un tensor al dispositivo especificado (CPU/GPU)"""
    device = device or DEVICE
    return tensor.to(device)

