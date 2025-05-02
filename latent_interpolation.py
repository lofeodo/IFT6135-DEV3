import torch
from torchvision.utils import save_image
from q1_train_vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('results/q1/model.pt', map_location=device)
model.eval()

def generate_interpolations():
    """
    Generate and visualize interpolations between two random points in latent space
    and their corresponding pixel space interpolations.
    """
    with torch.no_grad():
        z0 = torch.randn(1, 20).to(device)
        z1 = torch.randn(1, 20).to(device)
        
        x0 = model.decode(z0)
        x1 = model.decode(z1)
        
        alphas = torch.linspace(0, 1, 11)
        
        latent_interpolations = []
        pixel_interpolations = []
        
        for alpha in alphas:
            z_alpha = alpha * z0 + (1 - alpha) * z1
            x_alpha = model.decode(z_alpha)
            latent_interpolations.append(x_alpha.view(1, 28, 28))
            
            x_hat_alpha = alpha * x0 + (1 - alpha) * x1
            pixel_interpolations.append(x_hat_alpha.view(1, 28, 28))
        
        latent_interpolations = torch.cat(latent_interpolations, dim=0)
        pixel_interpolations = torch.cat(pixel_interpolations, dim=0)
        
        latent_interpolations = latent_interpolations.view(-1, 1, 28, 28)
        pixel_interpolations = pixel_interpolations.view(-1, 1, 28, 28)
        
        save_image(latent_interpolations, 'results/q1/q9_latent_interpolation.png', nrow=11, normalize=True)
        save_image(pixel_interpolations, 'results/q1/q9_pixel_interpolation.png', nrow=11, normalize=True)
        

if __name__ == "__main__":
    print("Generating interpolations...")
    generate_interpolations()
    print("Interpolations saved to 'results/q1/q9_latent_interpolation.png' and 'results/q1/q9_pixel_interpolation.png'") 