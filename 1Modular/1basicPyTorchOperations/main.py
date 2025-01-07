from utils.device_utils import move_to_device
from utils.tensor_ops import  double_tensor

from config import DEVICE   
import torch
from icecream import ic




def main():
    #create tensor
    x=torch.tensor([1.0,2.0,3.0])
    ic(x)


    #move tensor 
    x_device=move_to_device(x)

    ic(x_device, DEVICE)  

    # Realizar una operación en el tensor
    y_device = double_tensor(x_device)
    ic(y_device)  

    y_cpu = move_to_device(y_device, device='cpu')
    ic(y_cpu)  



if __name__=="__main__":
    main()

#