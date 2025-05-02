import torch
from torchvision.utils import save_image
from q1_train_vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('results/q1/model.pt', map_location=device)
model.eval()

def generate_samples(num_samples=64):
    with torch.no_grad():
        z = torch.randn(num_samples, 20).to(device)
        samples = model.decode(z)
        
        samples = samples.view(num_samples, 1, 28, 28)
        save_image(samples, 'results/q1/q7_generated_samples.png', nrow=8, normalize=True)

if __name__ == "__main__":
    print("Generating samples...")
    generate_samples()
    print("Samples saved to 'results/q1/q7_generated_samples.png'") 