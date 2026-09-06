import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib.gridspec as gridspec

import matplotlib.patches as mpatches
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from matplotlib.colors import to_rgb
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import csv
import os


def pairwise_distances(x):
    # cosine distances or Euclidean
    x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = x_norm @ x_norm.t()
    cos_dist = 1 - cos_sim
    return cos_dist


def pairwise_angular(z):
    # z should already be on sphere, but normalize just in case
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = z @ z.t()
    # angular distance = arccos of cosine similarity
    ang = torch.acos(torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6))
    return ang


def distance_preserving_loss(x, z, zmode="angular"):
    Dx = pairwise_distances(x)
    if zmode == "angular":
        Dz = pairwise_angular(z)
    else:
        Dz = pairwise_distances(z)

    # vectorize upper triangle to avoid trivial diagonal terms
    mask = torch.triu(torch.ones_like(Dx), diagonal=1).bool()

    Dx_vec = Dx[mask]
    Dz_vec = Dz[mask]

    # Pearson correlation (scale-invariant)
    Dx_norm = (Dx_vec - Dx_vec.mean()) / (Dx_vec.std() + 1e-8)
    Dz_norm = (Dz_vec - Dz_vec.mean()) / (Dz_vec.std() + 1e-8)

    corr = (Dx_norm * Dz_norm).mean()

    return 1 - corr  # maximize correlation → minimize 1 - corr


def tsne_loss(x, z, sigma=1.0, eps=1e-8):
    # pairwise Euclidean distances
    Dx = torch.cdist(x, x)
    Dz = torch.cdist(z, z)

    # Compute high-dimensional affinities (Gaussian kernel)
    Px = torch.exp(-Dx**2 / (2 * sigma**2))
    Px.fill_diagonal_(0)
    Px = Px / (Px.sum(dim=1, keepdim=True) + eps)  # normalize row-wise
    Px = (Px + Px.T) / 2  # symmetrize

    # Compute low-dimensional affinities (Student t kernel)
    Qz = 1 / (1 + Dz**2)
    Qz.fill_diagonal_(0)
    Qz = Qz / (Qz.sum() + eps)

    # Kullback-Leibler divergence between distributions
    loss = (Px * torch.log((Px + eps) / (Qz + eps))).sum()

    return loss


class NormalizedLinear(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, dim=-1)


class SmallAutoencoder(nn.Module):

    def __init__(self, input_dim=6, mode="linear"):
        super().__init__()

        self.mode = mode

        if self.mode == "linear":
            latent_dim = 2

        elif self.mode == "spherical":
            latent_dim = 3  # encoder outputs sphere (normalized later)

        elif self.mode == "lambert":
            latent_dim = 3  # encoder outputs 3D sphere, projected to 2D

        else:
            raise ValueError(f"Unknown mode: {mode}")

        act = nn.ReLU
        norm = nn.LayerNorm
        capacity = 16

        # ----------------------------
        # Encoder → latent_dim
        # ----------------------------
        self.encoder = nn.Sequential(nn.Linear(input_dim, capacity), act(),
                                     norm(capacity),
                                     nn.Linear(capacity, capacity * 2), act(),
                                     norm(capacity * 2),
                                     nn.Linear(capacity * 2, capacity * 4),
                                     norm(capacity * 4), act(),
                                     nn.Linear(capacity * 4, capacity),
                                     norm(capacity), act(),
                                     nn.Linear(capacity, latent_dim))

        if self.mode in ["spherical", "lambert"]:
            self.encoder = nn.Sequential(self.encoder, NormalizedLinear())

        # ----------------------------
        # Decoder (takes latent_dim=2 or 3 depending mode)
        # ----------------------------

        self.decoder = nn.Sequential(nn.Linear(latent_dim, capacity), act(),
                                     norm(capacity),
                                     nn.Linear(capacity, capacity * 4), act(),
                                     norm(capacity * 4),
                                     nn.Linear(capacity * 4, capacity * 2),
                                     norm(capacity * 2), act(),
                                     nn.Linear(capacity * 2, capacity),
                                     norm(capacity), act(),
                                     nn.Linear(capacity, input_dim))

    # ----------------------------------------------------
    # SPHERICAL UTILITIES
    # ----------------------------------------------------
    def get_polar(self, z3D):
        x, y, z = z3D[:, 0], z3D[:, 1], z3D[:, 2]
        r = torch.sqrt(x * x + y * y + z * z) + 1e-12
        theta = torch.acos(z / r)  # [0, π]
        phi = torch.atan2(y, x)  # [-π, π]
        return torch.stack([theta, phi], dim=-1)

    def get_sphere(self, z2D):
        theta, phi = z2D[:, 0], z2D[:, 1]
        sin_theta = torch.sin(theta)
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)

    # ----------------------------------------------------
    # LAMBERT AZIMUTHAL EQUAL-AREA
    # ----------------------------------------------------
    def lambert_forward(self, xyz):
        """ xyz: (N, 3) unit sphere → (N, 2) LAEA plane """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        k = torch.sqrt(2.0 / (1.0 + z + 1e-12))
        X = k * x
        Y = k * y
        return torch.stack([X, Y], dim=-1)

    def lambert_inverse(self, xy):
        """ xy: (N, 2) plane → (N, 3) sphere """
        X, Y = xy[:, 0], xy[:, 1]
        rho2 = X * X + Y * Y
        z = 1.0 - rho2 / 2.0
        z = torch.clamp(z, -1.0, 1.0)
        t = torch.sqrt(torch.clamp(1 - z * z, min=0.0))
        # Avoid NaN by normalizing direction
        denom = torch.sqrt(rho2) + 1e-12
        x = X / denom * t
        y = Y / denom * t
        return torch.stack([x, y, z], dim=-1)

    # ----------------------------------------------------
    # FORWARD / ENCODE / DECODE
    # ----------------------------------------------------
    def forward(self, x):
        lat = self.encoder(x)
        rec = self.decoder(lat)
        return rec, lat

    def encode(self, x):
        lat = self.encoder(x)

        if self.mode == "lambert":
            lat = self.lambert_forward(lat)
            lat = lat / 2
            return lat

        elif self.mode == "spherical":
            lat = self.get_polar(lat)
            lat[:, 0] = lat[:, 0] / (torch.pi / 2) - 1.
            lat[:, 1] = lat[:, 1] / torch.pi
            return lat

        return lat

    def decode(self, x):
        if self.mode == "lambert":
            x = x * 2
            sphere = self.lambert_inverse(x)
            return self.decoder(sphere)

        elif self.mode == "spherical":
            x[:, 0] = (x[:, 0] + 1.) * (torch.pi / 2)
            x[:, 1] = x[:, 1] * torch.pi
            sphere = self.get_sphere(x)
            return self.decoder(sphere)

        else:
            return self.decoder(x)


def regularization_loss(latent_batch):
    """
    Takes a (B, 2) tensor of latent vectors and returns a scalar loss.
    Replace the body of this function with your custom regularization.
    """
    loss = torch.relu(torch.abs(latent_batch) - 0.8).mean()
    return loss


def uniformity_loss(z, margin=1.):
    # Assume z in ℝ², shape [B, 2]
    # Centering

    bound_loss = F.relu(z.abs() - margin).mean()

    z = z - z.mean(dim=0, keepdim=True)

    # Encourage spread within [-margin, margin]
    var_loss = (z.var(dim=0) -
                (margin**2) / 3)**2  # uniform variance = a²/3 for [-a,a]

    # Penalize out-of-bounds

    return var_loss.mean() + 5 * bound_loss


def train_autoencoder(embeddings,
                      num_steps=2000,
                      batch_size=32,
                      lr=1e-3,
                      val_split=0.2,
                      device="cpu",
                      mode="linear"):

    print(mode)

    model = SmallAutoencoder(input_dim=embeddings.shape[1],
                             mode=mode).to(device)
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
            recloss = 2 * criterion(reconstructed, inputs)

            if mode == "linear":
                regul_loss = 5 * uniformity_loss(latent)

                regul_loss += 0.005 * tsne_loss(inputs, latent)

            elif "spherical" in mode or "lambert" in mode:
                regul_loss = 5. * torch.exp(-2 * (
                    (latent.unsqueeze(1) - latent.unsqueeze(0))**
                    2).sum(-1)).mean()
            loss = recloss + regul_loss

            loss.backward()
            optimizer.step()

            step += 1
            stepbar.update(1)

            # Validation check
            if step % 100 == 0:
                model.eval()
                with torch.no_grad():
                    val_recon, val_latent = model(val_tensor)
                    val_loss = criterion(
                        val_recon,
                        val_tensor)  #+ regularization_loss(val_latent)

                stepbar.set_description(
                    f"Step {step} | Train: {recloss.item():.4f} | Val: {val_loss.item():.4f}"
                )

            if step >= num_steps:
                break
    model.eval()
    return model


def prepare_training(encoder,
                     post_encoder,
                     dataset,
                     num_examples,
                     device="cpu",
                     mode="dataset"):
    print("Building dataset")

    allzsem = []
    alllabels = []
    N_SIGNAL = 128

    # for idx in tqdm(range(len(dataset))):
    if num_examples is not None:
        num_examples_per_ds = num_examples // len(dataset.datasets)
    else:
        num_examples_per_ds = None

    for name, curdataset in dataset.datasets.items():
        if num_examples_per_ds is None or num_examples_per_ds > len(
                curdataset):
            indexes = np.arange(len(curdataset))
        else:
            indexes = np.random.choice(len(curdataset),
                                       num_examples_per_ds,
                                       replace=False)
        for idx in tqdm(indexes):
            data = curdataset[idx]
            z = data["z"][..., :N_SIGNAL]
            z = torch.from_numpy(z).unsqueeze(0).float().to(device)

            zsem = encoder(z)
            if post_encoder is not None:
                zsem = post_encoder(zsem)

            if mode == "dataset":
                label = name
            elif mode == "file":
                label = data["metadata"]["path"]
            else:
                label = "None"

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
        # bg_rgb = np.array(to_rgb("#3c3c3c"))
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

    import numpy as np
    from scipy.ndimage import gaussian_filter
    from matplotlib.colors import to_rgb

    def additive_blend_blur(ax, data, labels, cmap_list, bins, sigma, gamma,
                            brightness_scale):
        """
        Blend rule:
        1) Build per-class blurred density maps B_i.
        2) Total density D = sum_i B_i.
        3) Per-pixel class weights W_i = B_i / (D + eps).
        4) Opacity A = 1 - exp(-k * D_norm), where D_norm = D / D.max().
        5) Foreground color = class-weighted mix with subtle desaturation/darkening.
        6) Final = (1 - A) * BG + A * Foreground.
        7) Optional density-driven darkening toward a floor (no black crush).

        brightness_scale acts like k (opacity strength).
        """
        unique_labels = np.unique(labels)
        H = W = bins
        all_blurred = np.zeros((len(unique_labels), H, W), dtype=np.float32)
        x, y = data[:, 0], data[:, 1]

        # Canvas extents
        xmin, xmax = -1.2, 1.2
        ymin, ymax = -1.2, 1.2
        xedges = np.linspace(xmin, xmax, W + 1)
        yedges = np.linspace(ymin, ymax, H + 1)

        # 1) Per-class histograms -> blur -> gamma
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            xi, yi = x[mask], y[mask]
            hist, _, _ = np.histogram2d(xi, yi, bins=[xedges, yedges])
            if hist.sum() > 0:
                hist = hist / hist.sum()
            blurred = gaussian_filter(hist, sigma=sigma)
            all_blurred[i] = np.power(blurred.T, gamma)  # (H, W)

        # 2) Total density and normalized weights
        D = all_blurred.sum(axis=0)  # (H, W)
        D_max = float(D.max())
        if D_max <= 0:
            D_max = 1.0
        D_norm = D / D_max
        eps = 1e-8
        Wcls = all_blurred / (D[None, ...] + eps
                              )  # (C, H, W), sums to 1 where D>0

        # 3) Class color mix (foreground only)
        colors = np.array(cmap_list)[:, :3]  # (C, 3)
        mixed = np.einsum('chw,ck->hwk', Wcls, colors)  # (H, W, 3)

        # --- Foreground-only styling (no background impact) ---
        # Slightly reduce saturation and brightness of the foreground colors
        # --- Foreground-only styling (before compositing) ---
        saturation_scale = 0.5  # almost full saturation (less “pastel”)
        # no extra foreground gamma darkening
        brightness_gain = 1.8
        gamma_dark = 1.00

        gray = mixed.mean(axis=2, keepdims=True)
        mixed = gray + saturation_scale * (mixed - gray)
        mixed = np.clip(mixed, 0.0, 1.0)

        # --- Opacity: make colors own dense regions a bit more ---
        A_col = 1.0 - np.exp(-(brightness_scale * 1.4) * (D_norm**0.85))

        bg_rgb = np.array(to_rgb("#bcbcbc"), dtype=np.float32)
        # bg_rgb = np.array(to_rgb("#3c3c3c"), dtype=np.float32)

        # --- Foreground contrast boost toward bg (punchier without touching bg) ---
        F = mixed
        F_rel = F - bg_rgb  # vector from bg to fg
        contrast = 1.5  # ↑ for more punch (1.3–2.0 is sane)
        F_rel = np.tanh(contrast * F_rel) / np.tanh(contrast)
        F_rel = F_rel * brightness_gain  # tiny dim to avoid “fluo”
        # no gamma here (keeps chroma vivid)
        F_toned = np.clip(bg_rgb + F_rel, 0.0, 1.0)

        # --- Composite ---
        image = (1.0 - A_col)[..., None] * bg_rgb + A_col[..., None] * F_toned

        # --- Bounded darkening (keep the moody look, avoid black holes) ---
        dark_strength = 1.1  # was higher → reduce washout
        dark_power = 4.  # darken mostly at peaks
        S_cap = 0.5  # NEVER darken more than this
        S = np.minimum(S_cap, dark_strength * (D_norm**dark_power))

        floor_ratio = 0.  # darker than bg, but not near black
        floor = bg_rgb * floor_ratio

        image = (1.0 - S)[..., None] * image + S[..., None] * floor
        image = np.clip(image, 0.0, 1.0)

        ax.imshow(image,
                  extent=[xmin, xmax, ymin, ymax],
                  origin='lower',
                  interpolation='bilinear')

    # ------------------------------------------------------------------------------
    # 1. Prepare data
    embedding_2d = embeddings

    # minx, maxx, miny, maxy = -latent_range, latent_range, -latent_range, latent_range

    # embedding_2d = np.c_[2 * (embedding_2d[:, 0] - minx) / (maxx - minx) - 1,
    #                      2 * (embedding_2d[:, 1] - miny) / (maxy - miny) - 1]

    background_color = "#bcbcbc"
    # background_color = "#3c3c3c"

    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    unique_labels = le.classes_
    base_cmap = plt.get_cmap('tab10', len(unique_labels))
    # colors = [base_cmap(i)[:3] for i in range(len(unique_labels))]

    base_colors = [
        #
        to_rgb('#9b59b6'),  # purple
        to_rgb('#FBC15E'),  # yellow
        to_rgb('#E24A33'),  #Strong red-orange
        to_rgb('#3498db'),  # blue
        to_rgb('#2ecc71'),  # green
        to_rgb('#1abc9c'),  # turquoise
        to_rgb('#e67e22'),  # orange
        to_rgb('#e74c3c'),  # vivid red
        to_rgb('#f39c12'),  # amber / warm yellow
        to_rgb('#16a085'),  # deep teal
        to_rgb('#2980b9'),  # darker blue
        to_rgb('#27ae60'),  # emerald
        to_rgb('#8e44ad'),  # dark violet
        to_rgb('#d35400'),  # burnt orange
        to_rgb('#c0392b'),  # strong crimson
        to_rgb('#7f8c8d'),  # grey (neutral tone)
        to_rgb('#95a5a6'),  # light grey
    ]
    base_colors = 8 * base_colors
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
            linewidths=0.04,
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
