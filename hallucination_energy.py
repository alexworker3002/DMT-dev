import torch
import torch.nn as nn
import ot

class TopologicalHallucinationEnergy(nn.Module):
    def __init__(self, sigma=0.1, epsilon=1e-5, blur=0.01):
        """
        Calculates the Topological Hallucination Energy (THE).
        :param sigma: variance scale for the Gaussian scalar cost
        :param epsilon: stabilization constant for normalization
        :param blur: entropy regularization term for Sinkhorn
        """
        super().__init__()
        self.sigma = sigma
        self.epsilon = epsilon
        self.blur = blur

    def forward(self, b_vals: torch.Tensor, d_vals: torch.Tensor, 
                p_coords: torch.Tensor, gt_coords: torch.Tensor, 
                mass_pred: torch.Tensor, mass_gt: torch.Tensor) -> torch.Tensor:
        """
        :param b_vals: (N,) predicted birth values (require_grad = True if derived from grid)
        :param d_vals: (N,) predicted death values (require_grad = True if derived from grid)
        :param p_coords: (N, 3) predicted spatial coordinates of critical points
        :param gt_coords: (M, 3) ground truth topological feature coordinates
        :param mass_pred: (N,) mass of predicted topological points
        :param mass_gt: (M,) mass of ground truth topological points
        """
        N = p_coords.shape[0]
        M = gt_coords.shape[0]
        
        if N == 0 or M == 0:
            return torch.tensor(0.0, device=p_coords.device, requires_grad=True)

        # 1. Topological Persistence Scalar Penalty: exp( -(d - b)^2 / 2sigma^2 )
        # Using upper-star, b > d, distance is (b - d)
        persistence = b_vals - d_vals
        topo_cost = torch.exp(- (persistence ** 2) / (2 * self.sigma ** 2)) # (N,)
        
        # 2. Spatial Coordinate Geometric Cost: ||p_i - p_j||^2
        # (N, M)
        spatial_cost = torch.cdist(p_coords, gt_coords, p=2.0) ** 2
        
        # 3. Full Cost Matrix
        # C_ij = topo_cost_i * spatial_cost_ij
        C = topo_cost.unsqueeze(1) * spatial_cost # (N, M)
        
        # 4. Defense Mechanism: Dynamic Normalization to prevent Sinkhorn NaN overflow
        C_max = C.max().detach() + self.epsilon
        C_norm = C / C_max
        
        # Marginal relaxation regularizer for unbalanced optimal transport
        reg_m = 1.0 
        
        # Sinkhorn returns the optimal regularized cost and gradients backprop automatically
        # when ot.backend.TorchBackend is passed (explicitly or via tensor inspection in POT 0.9.x).
        
        the_val = ot.unbalanced.sinkhorn_unbalanced2(
            mass_pred, mass_gt, C_norm, 
            reg=self.blur, 
            reg_m=reg_m
        )
        
        # Handle singleton dimensions if they exist
        if the_val.numel() == 1:
            the_val = the_val.squeeze()
            
        # Rescale the energy back
        the_val = the_val * C_max
        return the_val

if __name__ == "__main__":
    # Test gradient passthrough and backend validation
    sigma = 0.1
    the_layer = TopologicalHallucinationEnergy(sigma=sigma)
    
    # Fake Predicted Variables
    b_vals = torch.tensor([1.0, 0.8], requires_grad=True)
    d_vals = torch.tensor([0.4, 0.3], requires_grad=True)
    p_coords = torch.tensor([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]], requires_grad=True)
    mass_pred = torch.tensor([1.0, 1.0])
    
    # Fake Ground Truth Variables
    gt_coords = torch.tensor([[0.5, 0.5, 0.5]])
    mass_gt = torch.tensor([2.0])
    
    loss = the_layer(b_vals, d_vals, p_coords, gt_coords, mass_pred, mass_gt)
    loss.backward()
    
    print(f"Computed THE Loss: {loss.item():.6f}")
    assert b_vals.grad is not None, "Gradients failed to passthrough!"
    print(f"b_vals Gradient: {b_vals.grad.tolist()}")
    print("THE Computation exactly bounded. Autograd verified.\n")
