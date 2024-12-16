import torch
from model import VAE
from data import get_dataloader

def train(model, optimizer, dataloader, num_epochs):
    model.train()
 
    all_loss = {
        "loss": [],
        "Reconstruction_Loss": [],
        "KLD": []
    }
    for epoch in range(num_epochs):
        total_loss = {
            "loss": 0,
            "Reconstruction_Loss": 0,
            "KLD": 0
        }
        for batch in dataloader:
            sketch, _, _, _ = batch
            sketch = sketch.cuda()
            
            optimizer.zero_grad()
            recons, input, mu, log_var = model(sketch)
            
            loss = model.loss_function(recons, input, mu, log_var, M_N=0.00025)
            for key in total_loss:
                total_loss[key] += loss[key].item()
            
            loss["loss"].backward()
            optimizer.step()
                
        for key in loss:
            all_loss[key].append(total_loss[key] / len(dataloader))
        print(f"Epoch {epoch+1}/{num_epochs}, Average loss: {all_loss['loss'][-1]:.4f}, Reconstruction Loss: {all_loss['Reconstruction_Loss'][-1]:.4f}, KLD: {all_loss['KLD'][-1]:.4f}")

if __name__ == "__main__":
    img_size = 64
    root_dir = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong"
    model = VAE(img_size=img_size, in_channels=1, latent_dim=128).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    dataloader = get_dataloader(batch_size=256, num_samples=20000, img_size=img_size)
    train(model, optimizer, dataloader, num_epochs=100)

    torch.save(model.state_dict(), f'{root_dir}/vae_model.pth')