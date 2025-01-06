from utils.device_utils import move_to_device


from config import DEVICE   
import torch


def main():
    #create tensor
    x=torch.tensor([1.0,2.0,3.0])
    print("Tensor en CPU",x)


    #move tensor 
    x_device=move_to_device(x)

    print(f"Tensor en {DEVICE}:")


if __name__=="__main__":
    main()

#