import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm

# from torchsummary import summary

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def linear_sn(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

def conv2d_sn(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def batchNorm2d_sn(*args, **kwargs):
    return spectral_norm(nn.BatchNorm2d(*args, **kwargs))


class Gen_MLP(nn.Module):
    def __init__(self, zdim, out_shape):
        super().__init__()
        self.in_shape = (zdim,)
        self.out_shape = out_shape

        def _linear(chin, chout, bn):
            layers = [nn.Linear(chin, chout, bias=(not bn))]
            if (bn):
                layers.append(nn.BatchNorm1d(chout))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        use_batchnorm = True

        nin = zdim
        out_units = [256, 512, 1024]
        layers = []
        for nout in out_units:
            layers.append(_linear(nin, nout, bn=use_batchnorm))
            nin = nout

        # Final layer with TanH
        layers += [
            nn.Linear(nin, np.prod(out_shape)),
            nn.Tanh()
        ]

        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        # Get the in/out shapes
        B, _ = z.shape
        C, H, W = self.out_shape

        # Generate a new image via MLP
        out = self.mlp(z)

        # Reshape flat tensor into correct image shape
        out = out.view(B, C, H, W)
        return out


class Dis_MLP(nn.Module):
    def __init__(
        self,
        in_shape,
        use_spectral_norm=False
    ):
        super().__init__()
        self.in_shape = in_shape

        # Assemble the MLP
        nin = np.prod(in_shape)
        out_units = [1024, 512, 256, 1]
        layers = [nn.Flatten()]
        for nout in out_units:
            if (use_spectral_norm):
                layers += [linear_sn(nin, nout)]
            else:
                layers += [nn.Linear(nin, nout)]
            if (nout > 1):
                layers.append(nn.ReLU())
            nin = nout

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    

class Gen_WGAN(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.in_shape = (zdim,)

        def _convT2d(chin, chout, bn):
            if chin == 100:
                layers = [nn.ConvTranspose2d(chin, chout, 4, 1, 0, bias=(not bn))]
            else:
                layers = [nn.ConvTranspose2d(chin, chout, 4, 2, 1, bias=(not bn))]
            if (bn):
                layers.append(nn.BatchNorm2d(chout))
            layers.append(nn.ReLU(True))
            return nn.Sequential(*layers)

        use_batchnorm = True

        nin = zdim
        out_units = [512, 256, 128, 64]
        layers = []
        for nout in out_units:
            layers.append(_convT2d(nin, nout, bn=use_batchnorm))
            nin = nout

        # Final layer with TanH
        layers += [
            nn.ConvTranspose2d(nin, 3, 4, 2, 1),
            nn.Tanh()
        ]

        self.wgan = nn.Sequential(*layers)

    def forward(self, z):
        # Generate a new image via WGAN
        return self.wgan(z)


class Dis_WGAN(nn.Module):
    def __init__(self):
        super().__init__()

        def _conv2d(chin, chout, bn):
            layers = [conv2d_sn(chin, chout, 4, 2, 1, bias=(not bn))]
            if (bn):
                layers.append(batchNorm2d_sn(chout))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        use_batchnorm = True

        nin = 3
        out_units = [64, 128, 256, 512]
        layers = []
        for nout in out_units:
            layers.append(_conv2d(nin, nout, bn=use_batchnorm))
            nin = nout

        # Final layer
        layers += [
            conv2d_sn(nin, 1, 4, 1, 0),
        ]

        self.wgan = nn.Sequential(*layers)


    def forward(self, x):
        out = self.wgan(x)
        return torch.flatten(out)


def create_models(args, shape, device):

    use_sn = False

    # Spectral normalization with hinge loss
    if (args.loss == "hinge"):
        use_sn = True

    if args.loss == "wgan-gp":
        result = {
            # MLP generator
            "gen":  Gen_WGAN(zdim=args.zdim),

            # MLP discriminator
            "dis":  Dis_WGAN()
        }
    else:
        result = {
            # MLP generator
            "gen":  Gen_MLP(zdim=args.zdim, out_shape=shape),

            # MLP discriminator
            "dis":  Dis_MLP(in_shape=shape, use_spectral_norm=use_sn)
        }

    # Move the models to the GPU and print their stats
    for k, v in result.items():
        v.train()
        result[k] = v.to(device)

        print("Summary for: ", k)
        print(result[k])
        # summary(result[k], input_size=result[k].in_shape)
        print("")

    return result


def create_eval(args):
    return Gen_WGAN(zdim=args.zdim)
