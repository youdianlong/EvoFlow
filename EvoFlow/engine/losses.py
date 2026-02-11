import torch
import torch.nn.functional as F


def standard_flow_loss(model, x_target, history, cond, P_mean=-0.8, P_std=1.2):
    B = x_target.shape[0]
    device = x_target.device

    if history.dim() == 6:
        x_last = history[:, -1]
    elif history.dim() == 5:
        x_last = history[:, -1].unsqueeze(1)
    else:
        raise ValueError(f"Unexpected history shape: {history.shape}")

    x0 = torch.randn_like(x_target)
    x1 = x_target - x_last

    t = torch.sigmoid(torch.randn(B, device=device) * P_std + P_mean)
    t = t.clamp(0.01, 0.99)
    t_exp = t[:, None, None, None, None]

    z_t = t_exp * x1 + (1 - t_exp) * x0

    v_target = x1 - x0

    v_pred = model(z_t, t, cond, history)

    v_loss = F.mse_loss(v_pred, v_target)

    x1_pred = z_t + (1 - t_exp) * v_pred
    recon_loss = F.l1_loss(x1_pred, x1)

    pred_magnitude = x1_pred.abs().mean()
    target_magnitude = x1.abs().mean().clamp(min=1e-6)
    magnitude_loss = F.relu(pred_magnitude - target_magnitude * 2)

    total_loss = v_loss + 0.1 * recon_loss + 0.05 * magnitude_loss

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        total_loss = torch.tensor(1.0, device=device, requires_grad=True)

    metrics = {
        'v_loss': v_loss.item() if not torch.isnan(v_loss) else 0.0,
        'recon_loss': recon_loss.item() if not torch.isnan(recon_loss) else 0.0,
        'magnitude_loss': magnitude_loss.item() if not torch.isnan(magnitude_loss) else 0.0,
        'delta_mean': x1.abs().mean().item(),
        'delta_std': x1.std().item(),
        'pred_delta_mean': x1_pred.detach().abs().mean().item(),
        'v_pred_norm': v_pred.detach().norm().item() / B,
        'v_target_norm': v_target.norm().item() / B,
    }

    return total_loss, metrics
