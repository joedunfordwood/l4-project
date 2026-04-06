import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch.nn.functional as F

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Plots
def plot_loss_history(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sloc_loss_history(train_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def overlay_attr_on_grid(signals, attr_map, title='Test', cmap="hot", alpha=0.6, smooth_sigma=None, clip_percentile=None, figsize=(30,20)):
    signals = np.asarray(signals)
    attr = np.asarray(attr_map)


    leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    # normalise globally to keep color scale consistent across leads
    attrn = heatmap_normalise_attr(attr, clip_percentile=clip_percentile)

    fig, axis = plt.subplots(12, 1, sharex=True)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    fig.suptitle(title)



    im_handle = None

    n_samples=500

    for i in range(12):
        ax = axis[i]

        sig = signals[i]

        pad = 0.05 * (sig.max() - sig.min() + 1e-12)
        ymin = sig.min() - pad
        ymax = sig.max() + pad

        heat = attrn[i:i+1, :]  # shape (1, n_samples)
        extent = [0, n_samples, ymin, ymax]  # x0, x1, y0, y1

        im = ax.imshow(heat, aspect='auto', cmap=cmap, extent=extent,
                  origin='lower', alpha=alpha, interpolation='bilinear')

        im_handle = im

        # plot ECG waveform on top
        ax.plot(np.arange(n_samples), sig, color='black')

        ax.set_ylabel(leads[i], fontsize=9, rotation=0, labelpad=24)
        """
        # keep x-limits consistent
        ax.set_xlim(0, n_samples)
        # set y-limits to the computed range so heatmap aligns
        ax.set_ylim(ymin, ymax)"""

    plt.tight_layout()
    cbar = fig.colorbar(im_handle, ax=axis.ravel().tolist(), orientation='vertical', pad=0.02)
    cbar.set_label('Attribution')
    #plt.show()


    return fig

def print_report(result):
    sep = "=" * 60

    for lr in result["per_label"]:
        print(f"\n{sep}\n  {lr['name']}\n{sep}")

        print(f"\n  Samples  valid: {lr['valid_count']}  |  "
              f"NaN: {lr['nan_count']} ({lr['nan_pct']:.1f}%)")
        print(f"  Avg attribution map shape: {lr['avg_attr_map'].shape}  "
              f"(mean={lr['avg_attr_map'].mean():.5f})")

        ls = lr["lead_stats"]
        print(f"\n  Lead importance (mean ± std)")
        for rank, name in enumerate(ls["ranking"]):
            idx = LEAD_NAMES.index(name)
            print(f"{rank+1:2d}  {name:4s}  {ls['means'][idx]:.5f} ± {ls['stds'][idx]:.5f}")


    g = result["global"]
    print(f"\n{sep}\n  Global n{sep}")
    print(f"\n  Samples  valid: {g['valid_count']}  |  "
          f"NaN: {g['nan_count']} ({g['nan_pct']:.1f}%)")
    ls = g["lead_stats"]
    print(f"\n  Lead importance (mean ± std)")
    for rank, name in enumerate(ls["ranking"]):
        idx = LEAD_NAMES.index(name)
        print(f"    #{rank+1:2d}  {name:4s}  {ls['means'][idx]:.5f} ± {ls['stds'][idx]:.5f}")

def plot_results(result):
    labels = [lr["name"] for lr in result["per_label"]]
    n = len(labels)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    #  Per-lead attribution per label
    fig1, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(LEAD_NAMES))
    width = 0.8 / n
    for i, lr in enumerate(result["per_label"]):
        ls = lr["lead_stats"]
        ax1.bar(x + (i - n / 2 + 0.5) * width, ls["means"], width,
                 capsize=3,
                label=lr["name"], color=colors[i], alpha=0.85, edgecolor="white")
    gls = result["global"]["lead_stats"]
    ax1.plot(x, gls["means"], color="black", marker="o", markersize=4,
             linewidth=1.2, linestyle="--", label="global mean", zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LEAD_NAMES)
    ax1.set_title("Per-lead mean attribution by label")
    ax1.set_ylabel("Mean attribution")
    ax1.legend(fontsize=9, ncol=n + 1)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig1.tight_layout()

    # Superpixel heatmap
    sp_matrix = np.stack([lr["superpixel_stats"]["global_means"]
                          for lr in result["per_label"]])
    num_sp  = sp_matrix.shape[1]
    sp_size = result["per_label"][0]["superpixel_stats"]["sp_size"]
    tick_step = max(1, num_sp // 10)
    tick_pos  = np.arange(0, num_sp, tick_step)

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    im = ax2.imshow(sp_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest", vmax=0.00025, vmin=0.00007)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([str(i * sp_size) for i in tick_pos], fontsize=8)
    ax2.set_xlabel(f"Time step  (superpixel size = {sp_size})")
    ax2.set_title("Superpixel attribution heatmap (global across leads)")
    fig2.colorbar(im, ax=ax2, label="Mean attribution", pad=0.02)
    fig2.tight_layout()

    #  NaN summary
    nan_counts   = [lr["nan_count"]   for lr in result["per_label"]]
    nan_pcts     = [lr["nan_pct"]     for lr in result["per_label"]]
    valid_counts = [lr["valid_count"] for lr in result["per_label"]]

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4))
    ax3a.bar(labels, valid_counts, label="Valid", color="#4C9BE8", alpha=0.85, edgecolor="white")
    ax3a.bar(labels, nan_counts,   label="NaN",   color="#E84C4C", alpha=0.85,
             bottom=valid_counts, edgecolor="white")
    ax3a.set_title("Sample counts per label")
    ax3a.set_ylabel("Number of samples")
    ax3a.legend(fontsize=9)

    ax3b.bar(labels, nan_pcts, color="#E84C4C", alpha=0.85, edgecolor="white")
    ax3b.set_title("NaN percentage per label")
    ax3b.set_ylabel("NaN %")
    ax3b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    g = result["global"]
    ax3b.axhline(g["nan_pct"], color="black", linestyle="--", linewidth=1,
                 label=f"overall {g['nan_pct']:.1f}%")
    ax3b.legend(fontsize=9)
    fig3.suptitle(f"NaN summary: {g['nan_count']} / {g['valid_count'] + g['nan_count']} "
                  f"total removed ({g['nan_pct']:.1f}%)")
    fig3.tight_layout()

    #  Avg attribution map overlays
    overlay_figs = []

    for lr in result["per_label"]:
        fig = overlay_attr_on_grid(signals=np.zeros((12,500)), attr_map=lr["avg_attr_map"], title=f"{lr['name']}- Average attribution map",cmap="hot", alpha=0.7,clip_percentile=(1,99))
        overlay_figs.append(fig)

    fig_global = overlay_attr_on_grid(signals=np.zeros((12,500)), attr_map=result["global"]["avg_attr_map"],  title="Overall average attribution map",cmap="hot", alpha=0.7)


    plt.show()
    return fig1, fig2, fig3, overlay_figs, fig_global

def plot_discriminability(result, disc_result):
  FIG_WIDTH = 30
  labels = [lr['name'] for lr in result['per_label']]
  n = len(labels)

  n_samples = disc_result['cov_map'].shape[1]

  sig = disc_result['cov_map']

  # Coefficient of variation map
  fig1, ax1 = plt.subplots(figsize=(14, 3))
  #fig1.set_figwidth(FIG_WIDTH)
  im = ax1.imshow(heatmap_normalise_attr(disc_result['cov_map'],(1,99)), aspect='auto', cmap='magma', interpolation='bilinear')
  ax1.set_yticks(range(12))
  ax1.set_yticklabels(LEAD_NAMES, fontsize=8)
  ax1.set_xlabel('Time sample')
  ax1.set_title('Cross-label coefficient of variation  (brighter = more discriminative)')
  fig1.colorbar(im, ax=ax1, orientation='vertical', pad=0.02, label='CoV')

  # Differentiability map- deviation from global mean
  diff = disc_result['diff_maps']
  vabs = np.abs(diff).max()

  fig2, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)

  if n == 1:
    axes = [axes]

  for ax, label, d in zip(axes, labels, diff):
    im = ax.imshow(d, aspect='auto', cmap='RdBu_r',vmin=-vabs, vmax=vabs, interpolation='bilinear')

    ax.set_yticks(range(12))
    ax.set_yticklabels(LEAD_NAMES, fontsize=8)
    ax.set_title(f'{label}  (deviation from global mean)', fontsize=10)

  axes[-1].set_xlabel('Time sample')
  fig2.colorbar(im, ax=axes, orientation='vertical', pad=0.02, label='Attribution deviation')
  fig2.suptitle('Class-specific attribution  (red = above global avg, blue = below)', fontsize=11)

  plt.show()
  return fig1, fig2


def iii_heatmap_comparison(result):

  LEAD = 'III'

  n = len(result['per_label'])

  fig, axis = plt.subplots(n, 1, figsize=(14, 1*n), sharex=True)
  fig.suptitle("Lead III attribution map comparison")

  for i in range(n):

    ax = axis[i]
    heat = heatmap_normalise_attr(result['per_label'][i]['avg_attr_map'][None, 2], clip_percentile=(1,99))

    im = ax.imshow(heat, aspect='auto', cmap='hot', alpha=0.8, interpolation='bilinear')
    ax.set_ylabel(f"{result['per_label'][i]['name']}", fontsize=9)

  cbar = fig.colorbar(im, ax=axis, orientation='vertical', pad=0.02)
  cbar.set_label('Attribution')
  #plt.tight_layout()
  plt.show()

  return fig


# Misc functions
def seed_everything(seed=42):
    import random, os, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalise_attr(attr, score):

    attr = torch.sigmoid(attr)

    return attr

def filter_nan(block):
    nan_mask = np.isnan(block).all(axis=(1, 2))
    return block[~nan_mask], int(nan_mask.sum()), nan_mask

def get_weighting(dataset):
  _, counts = np.unique(dataset, return_counts=True)

  total = sum(counts)

  weighting = torch.Tensor([i/total for i in np.unique(dataset, return_counts=True)[1]])
  return weighting

def heatmap_normalise_attr(attr, clip_percentile=None):
    a = np.array(attr, dtype=np.float32)
    if clip_percentile is not None:
        lo, hi = np.percentile(a, clip_percentile[0]), np.percentile(a, clip_percentile[1])
        a = np.clip(a, lo, hi)

    a = (a - a.min()) / (a.max() - a.min() + 1e-12)

    return a

def _batch_eval_probs(model, inputs, device, target_class, batch_size=64):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, inputs.shape[0], batch_size):
            b = inputs[i:i+batch_size].to(device)
            logits = model(b)
            p = F.softmax(logits, dim=1)[:, target_class].cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0)

def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            all_preds.append(outputs.cpu())
            all_labels.append(y_batch)
            del x_batch, y_batch, outputs
    return torch.cat(all_preds), torch.cat(all_labels)