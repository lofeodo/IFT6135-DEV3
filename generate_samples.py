import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from q1_train_vae import VAE  # Import the VAE class from the training script

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = torch.load('results/q1/model.pt', map_location=device)
model.eval()  # Set to evaluation mode

def generate_samples(num_samples=64):
    with torch.no_grad():
        # Generate random latent vectors from standard normal distribution
        z = torch.randn(num_samples, 20).to(device)
        # Decode the latent vectors to generate images
        samples = model.decode(z)
        
        # Reshape and save the samples
        samples = samples.view(num_samples, 1, 28, 28)
        save_image(samples, 'results/q1/q7_generated_samples.png', nrow=8, normalize=True)
        
        # Also generate reconstructions of some test images
        from torchvision import datasets, transforms
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=8, shuffle=True)
        
        test_batch, _ = next(iter(test_loader))
        test_batch = test_batch.to(device)
        recon_batch, _, _ = model(test_batch)
        
        # Save original and reconstructed images side by side
        comparison = torch.cat([test_batch[:8], recon_batch.view(8, 1, 28, 28)])
        save_image(comparison, 'results/q1/q7_reconstruction.png', nrow=8, normalize=True)

if __name__ == "__main__":
    print("Generating samples...")
    generate_samples()
    print("Samples saved to 'results/q1/q7_generated_samples.png' and 'results/q1/q7_reconstruction.png'") 