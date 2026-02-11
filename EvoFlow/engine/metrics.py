import torch


def calc_psnr_3d(x, y):
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def calc_ssim_3d(x, y):
    C1, C2 = 0.01**2, 0.03**2
    if x.dim() == 4:
        x = x.unsqueeze(0)
    if y.dim() == 4:
        y = y.unsqueeze(0)

    mu_x, mu_y = x.mean(), y.mean()
    sigma_x_sq, sigma_y_sq = x.var(), y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim.item()
