import numpy as np


def save_comparison_3d(history, generated, gt, save_path, metrics=None, slice_idx=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def to_np(x):
        return ((x + 1) / 2).clamp(0, 1).squeeze().cpu().numpy()

    if slice_idx is None:
        D = generated.shape[-3]
        slice_idx = D // 2

    T = history.shape[0]
    fig, axes = plt.subplots(2, max(4, T + 1), figsize=(4 * max(4, T + 1), 8))

    for t in range(T):
        hist_slice = to_np(history[t])[slice_idx]
        axes[0, t].imshow(hist_slice, cmap='gray', vmin=0, vmax=1)
        axes[0, t].set_title(f'History t-{T-t}')
        axes[0, t].axis('off')

    gen_slice = to_np(generated)[slice_idx]
    axes[0, T].imshow(gen_slice, cmap='gray', vmin=0, vmax=1)
    axes[0, T].set_title('Generated')
    axes[0, T].axis('off')

    for t in range(T + 1, axes.shape[1]):
        axes[0, t].axis('off')

    gt_slice = to_np(gt)[slice_idx]
    axes[1, 0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')

    diff = np.abs(gen_slice - gt_slice)
    axes[1, 1].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 1].set_title('|Gen - GT|')
    axes[1, 1].axis('off')

    last_hist_slice = to_np(history[-1])[slice_idx]
    residual = gen_slice - last_hist_slice
    axes[1, 2].imshow(residual, cmap='RdBu', vmin=-0.2, vmax=0.2)
    axes[1, 2].set_title('Predicted Change')
    axes[1, 2].axis('off')

    if metrics:
        info_str = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        axes[1, 3].text(0.5, 0.5, info_str, ha='center', va='center',
                       fontsize=12, transform=axes[1, 3].transAxes)
        axes[1, 3].axis('off')

    for t in range(4 if metrics else 3, axes.shape[1]):
        axes[1, t].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
