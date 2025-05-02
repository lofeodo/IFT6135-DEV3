from ddpm_utils.unet import UNet, load_weights
from ddpm_utils.args import *
from ddpm_utils.dataset import *
from q2_ddpm import *
from q2_trainer_ddpm import *
import os
import matplotlib.pyplot as plt

# Create directories for saving results
save_path = os.path.join(os.getcwd(), "results", "q2", "DDPM")
os.makedirs(save_path, exist_ok=True)

def generate_intermediate_images(trainer):
    """Generate and save intermediate samples from the diffusion process"""
    steps_to_show = [0, 100, 500, 800, 900, 950, 980, 999]
    images = trainer.generate_intermediate_samples(n_samples=4, steps_to_show=steps_to_show)
    return images, steps_to_show

def plot_intermediate_samples(images, steps_to_show, n_samples):
    """
    Plot the intermediate steps of the diffusion process
    Args:
        images: List of image tensors at different steps
        steps_to_show: List of steps that were captured
        n_samples: Number of images to show
    """
    plt.figure(figsize=(25, 15*n_samples))
    fig, axs = plt.subplots(n_samples, len(steps_to_show))
    
    for sample_idx in range(n_samples):
        for step_idx, img in enumerate(images):
            axs[sample_idx, step_idx].imshow(img[sample_idx, 0], cmap='gray')
            step = steps_to_show[step_idx] if step_idx < len(steps_to_show) else args.n_steps
            axs[sample_idx, step_idx].set_title(f'Image {sample_idx}\nt={args.n_steps - step-1}', size=8)
            axs[sample_idx, step_idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'intermediate_samples.png'))
    plt.close()

def main():
    eps_model = UNet(c_in=1, c_out=1)
    eps_model = load_weights(eps_model, args.MODEL_PATH)

    diffusion_model = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=args.n_steps,
        device=args.device
    )

    trainer = Trainer(args, eps_model, diffusion_model)

    dataloader = torch.utils.data.DataLoader(
        MNISTDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    # Train the model
    trainer.train(dataloader)
    
    # Generate and save intermediate samples
    print("Generating intermediate samples...")
    images, steps_to_show = generate_intermediate_images(trainer)
    plot_intermediate_samples(images, steps_to_show, n_samples=4)
    print("Saved intermediate samples")

if __name__ == "__main__":
    main()
