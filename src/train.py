import argparse
import wandb

import ganlib.loggers as l
from trainer import Trainer


def do_train(args):
    wandb.login()
    run = wandb.init(project="portrait-gan", id=args.name)

    trainer = Trainer(args)

    # Setup our logging
    trainer.setup(
        loggers=[
            l.WandbLogger(args=args),
            l.ConsoleLogger(interval=args.log_interval),
            # l.CheckpointMaker(trainer, interval=10*1000),
            l.ImageSampler(
                trainer,
                interval=args.log_interval,
                sinks=[
                    l.ImageFileSink(trainer, only_current=True),
                    l.ImageWandbSink(trainer)
                ],
                shape=(3, 6)
            )
        ]
    )

    # Commence training
    trainer.train()


if (__name__ == "__main__"):

    p = argparse.ArgumentParser()

    # Command line arguments
    p.add_argument("--name", "-n", default="",
                   type=str, help="Experiment name")
    p.add_argument("--outfolder", "-o", default=".scratch/runs",
                   type=str, help="Output folder for training runs")

    # Basic GAN settings
    p.add_argument("--loss", "-l", default="gan", type=str,
                   help="Loss function to use for training")
    p.add_argument("--dataset", "-d", default="portraits",
                   type=str, help="Dataset")
    p.add_argument("--channels", "-ch", default=3, type=str, help="Channels")
    p.add_argument("--image_size", "-is", default=64,
                   type=str, help="Image size")
    p.add_argument("--batch_size", "-b", default=32,
                   type=int, help="Batch size")
    p.add_argument("--iters", "-it", default=1000*1000,
                   type=int, help="Number of training iterations")
    p.add_argument("--zdim", "-z", default=100, type=int,
                   help="Latent space dimensionality")
    p.add_argument("--lr_gen", "-lrG", default=0.0001,
                   type=float, help="Generator learning rate")
    p.add_argument("--lr_dis", "-lrD", default=0.0002,
                   type=float, help="Discriminator learning rate")
    p.add_argument("--use_hinge", "-uh", action="store_true",
                   help="Use hinge loss in WGAN")

    # Logging
    p.add_argument("--log_interval", "-lit", default=1*1000,
                   type=int, help="Number of training iterations")

    # Execute training
    do_train(p.parse_args())
