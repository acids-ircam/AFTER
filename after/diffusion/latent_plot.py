import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib.gridspec as gridspec

import matplotlib.patches as mpatches
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from matplotlib.colors import to_rgb

torch.set_grad_enabled(True)
from sklearn.model_selection import train_test_split

from tqdm import tqdm


class SmallAutoencoder(nn.Module):

    def __init__(self, input_dim=6, latent_dim=2):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.GELU(),
                                     nn.Linear(16, 16), nn.GELU(),
                                     nn.Linear(16, latent_dim))
        # Decoder
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 8), nn.GELU(),
                                     nn.Linear(8, 16), nn.GELU(),
                                     nn.Linear(16, input_dim))

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def regularization_loss(latent_batch):
    """
    Takes a (B, 2) tensor of latent vectors and returns a scalar loss.
    Replace the body of this function with your custom regularization.
    """
    loss = torch.relu(torch.abs(latent_batch) - 1).mean()
    return loss


def train_autoencoder(embeddings,
                      num_steps=2000,
                      batch_size=32,
                      lr=1e-3,
                      val_split=0.2,
                      device="cpu"):
    model = SmallAutoencoder(input_dim=embeddings.shape[1],
                             latent_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Split train/val
    X_train, X_val = train_test_split(embeddings,
                                      test_size=val_split,
                                      random_state=42)
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32))
    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    step = 0
    print("Starting training")
    stepbar = tqdm(range(num_steps), desc="Training Autoencoder")

    while step < num_steps:
        for batch in dataloader:
            model.train()
            inputs = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed, latent = model(inputs)
            loss = criterion(reconstructed,
                             inputs) + regularization_loss(latent)
            loss.backward()
            optimizer.step()

            step += 1
            stepbar.update(1)

            # Validation check
            if step % (num_steps // 10) == 0 or step == 1:
                model.eval()
                with torch.no_grad():
                    val_recon, val_latent = model(val_tensor)
                    val_loss = criterion(
                        val_recon,
                        val_tensor) + regularization_loss(val_latent)

                stepbar.set_description(
                    f"Step {step} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}"
                )

            if step >= num_steps:
                break

    return model


def prepare_training(encoder, dataset, num_examples, device="cpu"):
    print("Building dataset")

    allzsem = []
    alllabels = []
    N_SIGNAL = 128

    # for idx in tqdm(range(len(dataset))):
    for name, curdataset in dataset.datasets.items():
        if num_examples is None or num_examples > len(curdataset):
            indexes = np.arange(len(curdataset))
        else:
            indexes = np.random.choice(len(curdataset),
                                       num_examples,
                                       replace=False)
        for idx in tqdm(indexes):
            data = curdataset[idx]
            z = data["z"][..., :N_SIGNAL]
            z = torch.from_numpy(z).unsqueeze(0).float().to(device)
            label = name
            zsem = encoder(z)

            zsem = zsem.detach().cpu().numpy().squeeze()
            allzsem.append(zsem)
            alllabels.append(label)

    allzsem = np.stack(allzsem)
    return allzsem, alllabels


def generate_plot(embeddings,
                  labels,
                  use_blur=True,
                  bins=100,
                  sigma=2.0,
                  gamma=1.0,
                  brightness_scale=5.0):

    # ------------------------------------------------------------------------------
    # Embedded plotting function from your style
    # ------------------------------------------------------------------------------
    def additive_blend_blur(ax, data, labels, cmap_list, bins, sigma, gamma,
                            brightness_scale):
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        H = W = bins
        all_blurred = np.zeros((n_labels, H, W))
        x, y = data[:, 0], data[:, 1]
        xmin, xmax = -1.2, 1.2
        ymin, ymax = -1.2, 1.2
        xedges = np.linspace(xmin, xmax, W + 1)
        yedges = np.linspace(ymin, ymax, H + 1)

        for i, label in enumerate(unique_labels):
            xi = x[labels == label]
            yi = y[labels == label]
            hist, _, _ = np.histogram2d(xi, yi, bins=[xedges, yedges])
            if hist.sum() > 0:
                hist /= hist.sum()
            blurred = gaussian_filter(hist, sigma=sigma)
            all_blurred[i] = blurred.T**gamma

        # image = np.ones((H, W, 3))  # white base

        bg_rgb = np.array(to_rgb("#bcbcbc"))  # background color as RGB
        image = np.ones((H, W, 3)) * bg_rgb  # colored canvas
        for i, color in enumerate(cmap_list):
            norm_blur = all_blurred[i]
            norm_blur = norm_blur / norm_blur.max() if norm_blur.max(
            ) > 0 else norm_blur
            for c in range(3):
                # image[:, :, c] -= np.clip(
                #     (1 - color[c]) * norm_blur * brightness_scale, 0, 1.)
                image[:, :, c] -= np.clip(
                    (1 - color[c]) * norm_blur * brightness_scale, 0,
                    bg_rgb[c] - 0.2)

        image = np.clip(image, 0, 1)

        ax.imshow(image,
                  extent=[xmin, xmax, ymin, ymax],
                  origin='lower',
                  interpolation='bilinear')

    # ------------------------------------------------------------------------------
    # 1. Prepare data
    embedding_2d = embeddings
    background_color = "#bcbcbc"

    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    unique_labels = le.classes_
    base_cmap = cm.get_cmap('tab10', len(unique_labels))
    # colors = [base_cmap(i)[:3] for i in range(len(unique_labels))]

    base_colors = [
        # to_rgb('#e67e22'),  # orange
        to_rgb('#E24A33'),  #Strong red-orange
        to_rgb('#3498db'),  # blue
        to_rgb('#FBC15E'),  # yellow
        to_rgb('#9b59b6'),  # purple
        # to_rgb('#1abc9c'),  # turquoise
        to_rgb('#2ecc71'),  # green
    ]
    colors = [base_colors[i][:3] for i in range(len(unique_labels))]

    # ------------------------------------------------------------------------------
    # 2. Set up figure
    FIG_W, FIG_H = 8, 6
    fig = plt.figure(figsize=(FIG_W, FIG_H),
                     facecolor=background_color,
                     constrained_layout=True)
    ax = fig.add_subplot(facecolor=background_color)
    ax.axis('off')
    point_colors = np.array([colors[i] for i in label_ids])
    if use_blur:
        additive_blend_blur(ax,
                            embedding_2d,
                            label_ids,
                            colors,
                            bins=bins,
                            sigma=sigma,
                            gamma=gamma,
                            brightness_scale=brightness_scale)

        ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            s=0.2,
            c=point_colors,
            #    cmap=base_cmap,
            linewidths=0.03,
            alpha=0.7,
            zorder=4,
            edgecolor='white')
    else:
        ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            s=8,
            c=point_colors,
            #    cmap=bas\e_cmap,
            linewidths=0,
            alpha=0.8)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False)

    for spine in ax.spines.values():
        spine.set_color('black')

    # ------------------------------------------------------------------------------
    # Cross design and center dots
    line_length = 1.5
    dot_size = 80
    ax.plot([-line_length / 2, line_length / 2], [-1, -1],
            color='black',
            linewidth=1.5,
            zorder=2,
            alpha=0.8)
    ax.plot([-line_length / 2, line_length / 2], [1, 1],
            color='black',
            linewidth=1.5,
            zorder=2,
            alpha=0.8)
    ax.plot([-1, -1], [-line_length / 2, line_length / 2],
            color='black',
            linewidth=1.5,
            zorder=2,
            alpha=0.8)
    ax.plot([1, 1], [-line_length / 2, line_length / 2],
            color='black',
            linewidth=1.5,
            zorder=2,
            alpha=0.8)
    ax.scatter(0, 1, color='black', s=dot_size, zorder=3)
    ax.scatter(-1, 0, color='black', s=dot_size, zorder=3)
    ax.scatter(1, 0, color='black', s=dot_size, zorder=3)
    ax.scatter(0, -1, color='black', s=dot_size, zorder=3)

    plt.show()

    # ------------------------------------------------------------------------------
    # Legend
    LEGEND_WIDTH = 2
    legend_fig = plt.figure(figsize=(LEGEND_WIDTH, FIG_H),
                            facecolor=background_color)
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    handles = [
        mpatches.Patch(color=colors[i], label=unique_labels[i])
        for i in range(len(unique_labels))
    ]
    legend_ax.legend(handles=handles,
                     loc='center',
                     frameon=False,
                     ncol=1,
                     fontsize=10,
                     handlelength=1.5,
                     handletextpad=0.5,
                     borderaxespad=0.0,
                     borderpad=0.0,
                     labelspacing=1.2)

    return fig, legend_fig
