import torch
from model import BSplineTrajectory2DCINN, fit_bspline_trajectory
from data_real import get_dataloader

def train(model, optimizer, dataloader, num_epochs):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            sketch, params, fitted_traj = batch
            sketch, params, fitted_traj = sketch.cuda(), params.cuda(), fitted_traj.cuda()
            
            optimizer.zero_grad()
            z, log_jac_det = model(sketch, params)
            
            # Corrected loss calculation
            nll = torch.mean(z**2) / 2 - torch.mean(log_jac_det) / (model.n_dim_total)
            
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 10.)
            optimizer.step()
            
            total_loss += nll.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average NLL: {avg_loss:.4f}")
        
        scheduler.step()

if __name__ == "__main__":
    root_dir = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong"
    num_control_points = 10
    model = BSplineTrajectory2DCINN(num_control_points=num_control_points, degree=3).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    dataloader = get_dataloader(batch_size=256, num_control_points=num_control_points)
    train(model, optimizer, dataloader, num_epochs=100)
    
    torch.save(model.state_dict(), f'{root_dir}/bspline_cinn_model_2D_CINN_real.pth')