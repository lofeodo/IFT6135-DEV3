import matplotlib.pyplot as plt

from cfg_utils.args import *
from cfg_utils.dataset import *
from cfg_utils.unet import *
from q3_cfg_diffusion import *
from q3_trainer_cfg import *

def main():
    dataloader = torch.utils.data.DataLoader(
        MNISTDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    eps_model = UNet_conditional(c_in=1, c_out=1, num_classes=10)

    diffusion_model = CFGDiffusion(
                eps_model=eps_model,
                n_steps=args.n_steps,
                device=args.device,
            )

    trainer = Trainer(args, eps_model, diffusion_model)

    trainer.train(dataloader)
        

if __name__ == "__main__":
    main()
