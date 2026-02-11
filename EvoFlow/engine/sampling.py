import torch


@torch.no_grad()
def standard_flow_sample(model, history, cond, num_steps=50, cfg_scale=2.0, device='cuda'):
    B = history.shape[0]

    if history.dim() == 6:
        x_last = history[:, -1]
    elif history.dim() == 5:
        x_last = history[:, -1].unsqueeze(1)
    else:
        raise ValueError(f"Unexpected history shape: {history.shape}")

    shape = x_last.shape
    z = torch.randn(shape, device=device)
    ts = torch.linspace(0, 0.998, num_steps + 1, device=device)

    for i in range(num_steps):
        t_cur = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_cur
        t_batch = torch.full((B,), t_cur, device=device)

        if cfg_scale > 1.0:
            v = model.forward_with_cfg(z, t_batch, cond, history, cfg_scale)
        else:
            v = model(z, t_batch, cond, history)

        z = z + dt * v

    delta_pred = z.clamp(-0.5, 0.5)
    return x_last + delta_pred
