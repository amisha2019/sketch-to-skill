import torch
from model import SimplifiedBSplineModel
from data import get_dataloader

def train(model, optimizer, dataloader, num_epochs):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)
    model.train()
    
    mse_loss_fn = torch.nn.MSELoss()  # Define MSE loss function
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            sketch, traj, params_Gt, fitted_traj = batch
            sketch, params_Gt = sketch.cuda(), params_Gt.cuda()  # Move data to GPU
            
            optimizer.zero_grad()
            
            # Forward pass: predict control points from the sketch
            predicted_params = model(sketch)
            
            # Compute MSE loss between predicted and ground truth control points
            loss = mse_loss_fn(predicted_params, params_Gt)
            
            # Backpropagate and update the model
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss (MSE): {avg_loss:.4f}")
        
        scheduler.step()

if __name__ == "__main__":
    root_dir = "/fs/nexus-scratch/amishab/Teacher_student_RLsketch/saved_models"
    num_control_points = 10
    model = SimplifiedBSplineModel(num_control_points=num_control_points).cuda()
    
    # Define optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Get dataloader (you might need to adjust paths or dataset generation logic)
    dataloader = get_dataloader(batch_size=256, num_samples=20000, num_control_points=num_control_points)
    
    # Train the model for 100 epochs
    train(model, optimizer, dataloader, num_epochs=100)
    
    # Save the trained model
    torch.save(model.state_dict(), f'{root_dir}/bspline_mlp_model_2D.pth')
