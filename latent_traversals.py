import torch
from torchvision.utils import save_image
from q1_train_vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('results/q1/model.pt', map_location=device)
model.eval()

def generate_latent_traversals(num_samples=5, epsilon=2.0):
    """
    Generate latent traversals for each dimension of the latent space.
    
    Args:
        num_samples: Number of samples per dimension
        epsilon: Range of perturbation for each dimension
    """
    with torch.no_grad():
        z_base = torch.randn(1, 20).to(device)
        
        grid = []
        
        for dim in range(20):
            row = []
            for i in range(num_samples):
                z = z_base.clone()
                z[0, dim] = z_base[0, dim] + epsilon * (i - num_samples//2) / (num_samples//2)
                sample = model.decode(z)
                row.append(sample.view(1, 28, 28))
            
            row = torch.cat(row, dim=0)
            grid.append(row)
        
        grid = torch.cat(grid, dim=0)
        
        grid = grid.view(-1, 1, 28, 28)
        
        save_image(grid, 'results/q1/q8_latent_traversals.png', nrow=num_samples, normalize=True)
        

if __name__ == "__main__":
    print("Generating latent traversals...")
    generate_latent_traversals()
    print("Latent traversals saved to 'results/q1/q8_latent_traversals.png'") 