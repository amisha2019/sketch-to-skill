import torch
import torch.nn as nn

class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, x, y):
        """
        Calculate Chamfer Distance
        x: (batch_size, n_points, 3)
        y: (batch_size, m_points, 3)
        """
        x = x.unsqueeze(2)  # (B, N, 1, 3)
        y = y.unsqueeze(1)  # (B, 1, M, 3)
        
        dist = torch.sum((x - y) ** 2, dim=-1)  # (B, N, M)
        
        min_dist_xy = torch.min(dist, dim=2)[0]  # (B, N)
        min_dist_yx = torch.min(dist, dim=1)[0]  # (B, M)
        
        chamfer_dist = torch.mean(min_dist_xy, dim=1) + torch.mean(min_dist_yx, dim=1)
        
        return chamfer_dist

class FrechetDistance(nn.Module):
    def __init__(self):
        super(FrechetDistance, self).__init__()

    def forward(self, x, y):
        """
        Calculate Fréchet Distance
        x: (batch_size, n_points, 3)
        y: (batch_size, m_points, 3)
        """
        batch_size, n, _ = x.shape
        m = y.shape[1]

        # Initialize distance matrix
        dist_matrix = torch.full((batch_size, n + 1, m + 1), float('inf'), device=x.device)
        dist_matrix[:, 0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = torch.sum((x[:, i-1] - y[:, j-1]) ** 2, dim=1)
                min_prev = torch.min(torch.min(dist_matrix[:, i-1, j],
                                               dist_matrix[:, i, j-1]),
                                     dist_matrix[:, i-1, j-1])
                dist_matrix[:, i, j] = torch.where(cost > min_prev, cost, min_prev)

        return dist_matrix[:, -1, -1]

# Test function
def test_distances():
    batch_size, n, m = 32, 100, 120
    x = torch.rand(batch_size, n, 3, requires_grad=True).cuda()
    y = torch.rand(batch_size, m, 3, requires_grad=True).cuda()

    print(x.requires_grad, y.requires_grad)

    # chamfer_dist = ChamferDistance().cuda()
    frechet_dist = FrechetDistance().cuda()

    # chamfer_result = chamfer_dist(x, y)
    frechet_result = frechet_dist(x, y)

    # print("Chamfer Distance:", chamfer_result)
    print("Fréchet Distance:", frechet_result)

    # Test backward propagation
    # loss = chamfer_result.mean()
    with torch.autograd.detect_anomaly():
        loss = frechet_result.mean()
        loss.backward()

    print("Backward pass successful")

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    test_distances()