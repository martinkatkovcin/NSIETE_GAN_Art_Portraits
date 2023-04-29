
import yaml
import os
from datetime import datetime
import math
import numpy as np
import torch


def initialize_experiment(args):

    '''
        Create a new name for our run.
        Prepare folder structure for it
    '''
    name = args.name

    if (name == ""):
        now = datetime.now()
        name = now.strftime("%Y%m%d-%H%M%S")

    run_folder = os.path.join(args.outfolder, name)
    os.makedirs(run_folder, exist_ok=True)

    # Output the config parameters
    config_string = yaml.dump(vars(args))
    
    print("-----[ Training with parameters ]------")
    print(config_string)
    print("")

    fn = os.path.join(run_folder, "config.yaml")
    with open(fn, "w") as f:
        f.write(config_string)


    return name, run_folder



class RunningAverage:
    def __init__(self, decay=0.95):
        self.beta = decay
        self.reset()

    def reset(self):
        self.v = 0
        self.c = 0
        self.avg = 0

    def step(self, val):
        if (not math.isnan(val)):
            self.v = self.beta*self.v + (1.0 - self.beta)*val
            self.c = self.beta*self.c + (1.0 - self.beta)*1
            self.avg = self.v / self.c


class TrainingStats:
    def __init__(self, decay=0.95):
        self.lossG = RunningAverage(decay)
        self.lossD = RunningAverage(decay)

    def step(self, g, d):
        self.lossG.step(g)
        self.lossD.step(d)

        result = {
            "lossG": self.lossG.avg,
            "lossD": self.lossD.avg
        }

        return result


def to_image(x):    
    y = x.cpu().detach().numpy()
    y = np.clip(y, -0.5, 0.5)
    y = (y + 0.5) * 255.0
    y = y.astype(np.uint8)
    return y

def arange_images(x, shape):
    TH, TW = shape
    B,C,H,W = x.shape

    result = np.zeros(shape=(C,H*TH,W*TW), dtype=np.uint8)

    for i in range(TH):
        for j in range(TW):
            idx = i*TW + j
            result[:,(i+0)*H:(i+1)*H,(j+0)*W:(j+1)*W] = to_image(x[idx])

    # Transpose at the end if necessary
    if (C == 1):
        result = result[0]
    else:
        # Numpy images need to be in <H;W;C> shape 
        # when saving into file
        result = np.transpose(result, (1, 2, 0))

    return result


def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty
