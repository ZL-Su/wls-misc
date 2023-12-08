import numpy as np
import os.path, random
import torch
from torch.utils.data import Dataset

print("torch version {}".format(torch.__version__))


# get training data
dir = "./misc/pbdl/data/"
if True:
    if not os.path.isfile(dir + "data-airfoils.npz"):
        import requests

        print(
            "Downloading training data (300MB), this can take a few minutes the first time..."
        )
        with open(dir + "data-airfoils.npz", "wb") as datafile:
            resp = requests.get(
                "https://dataserv.ub.tum.de/s/m1615239/download?path=%2F&files=dfp-data-400.npz",
                verify=False,
            )
            datafile.write(resp.content)
    else:
        print("The dataset {} is already existed".format("data-airfoils.npz"))
