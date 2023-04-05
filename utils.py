import sys
import torch

# Get available device in this machine
def device():
   if(torch.cuda.is_available()):
      return torch.device("cuda")
   else:
      return torch.device("cpu")
   
def __init_env__(seed:int=42):
   torch.manual_seed(seed)
   str = "Using torch" + torch.__version__
   str = str + " on device " + device().type.capitalize()
   print(str)