import torch
from torchvision.utils import save_image
import numpy as np
from q1_train_vae import VAE

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = torch.load('results/q1/model.pt', map_location=device)
model.eval()  # Set to evaluation mode

def generate_latent_traversals(num_samples=5, epsilon=2.0):
    """
    Generate latent traversals for each dimension of the latent space.
    
    Args:
        num_samples: Number of samples per dimension
        epsilon: Range of perturbation for each dimension
    """
    with torch.no_grad():
        # Generate a base latent vector
        z_base = torch.randn(1, 20).to(device)
        
        # Create a grid to store all traversals
        grid = []
        
        # For each latent dimension
        for dim in range(20):
            row = []
            # Create perturbations for this dimension
            for i in range(num_samples):
                # Create a copy of the base vector
                z = z_base.clone()
                # Perturb the current dimension
                z[0, dim] = z_base[0, dim] + epsilon * (i - num_samples//2) / (num_samples//2)
                # Generate image
                sample = model.decode(z)
                row.append(sample.view(1, 28, 28))
            
            # Stack the row
            row = torch.cat(row, dim=0)
            grid.append(row)
        
        # Stack all rows
        grid = torch.cat(grid, dim=0)
        
        # Reshape the grid for proper visualization
        # The grid should be (num_samples * 20, 1, 28, 28)
        grid = grid.view(-1, 1, 28, 28)
        
        # Save the grid
        save_image(grid, 'results/q1/q7_latent_traversals.png', nrow=num_samples, normalize=True)
        
        # Print analysis
        print("\nLatent Space Analysis:")
        print("---------------------")
        print("The latent traversals show how each dimension affects the generated images.")
        print("A well-disentangled representation would show that each dimension controls")
        print("a distinct, interpretable feature of the generated digits (e.g., thickness,")
        print("rotation, position, etc.).")
        print("\nLook for:")
        print("1. Each row should show consistent changes in a specific feature")
        print("2. Different rows should control different features")
        print("3. The changes should be interpretable (e.g., digit thickness, slant, etc.)")
        print("\nThe saved image shows 20 rows (one per latent dimension) with 5 samples each.")
        print("Each row shows how varying that dimension affects the generated digit.")

if __name__ == "__main__":
    print("Generating latent traversals...")
    generate_latent_traversals()
    print("Latent traversals saved to 'results/q1/q8_latent_traversals.png'") 