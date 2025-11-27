import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_sbi_distributions():
    # --- 1. Setup Parameters ---
    # Prior: Theta ~ N(0, 2)
    mu_theta = 0.0
    sigma_theta = 1.0

    # Model: Parabolic likelihood x = theta^2 + noise
    sigma_x = 1.0

    # --- 2. Create Grid ---
    theta_range = np.linspace(-4, 4, 300)
    x_range = np.linspace(-2, 10, 300)
    Theta, X = np.meshgrid(theta_range, x_range)

    # --- 3. Calculate Densities ---

    # A. Prior P(theta)
    P_theta = norm.pdf(Theta, loc=mu_theta, scale=sigma_theta)

    # B. Likelihood P(x | theta)
    P_x_given_theta = norm.pdf(X, loc=(Theta**2), scale=sigma_x)

    # C. Joint P(x, theta) = P(x | theta) * P(theta)
    P_joint = P_x_given_theta * P_theta

    # D. Marginal P(x) - Numerical Integration
    P_x_marginal = np.trapz(P_joint, theta_range, axis=1).reshape(-1, 1)

    # E. Posterior P(theta | x) = P(x, theta) / P(x)
    epsilon = 1e-12
    P_theta_given_x = P_joint / (P_x_marginal + epsilon)

    # F. Ratio R(x, theta) = P(x, theta) / (P(x) * P(theta))
    # This is equivalent to P(x | theta) / P(x) or P(theta | x) / P(theta)
    P_marginal_product = (P_x_marginal + epsilon) * (P_theta + epsilon)
    P_ratio = P_joint / P_marginal_product

    # --- 4. Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    def setup_axis(ax, title, z_data, cmap="inferno"):
        # Rasterize contourf for efficient PDF saving
        # Removed colorbar creation
        contour = ax.contourf(Theta, X, z_data, levels=50, cmap=cmap)
        contour.set_rasterized(True)
        ax.set_xlabel(r"Parameter $\theta$", fontsize=12)
        ax.set_ylabel(r"Observation $x$", fontsize=12)
        ax.set_title(title, fontsize=13, pad=10, fontweight="bold")
        ax.set_ylim(x_range[0], x_range[-1])
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw the theoretical parabola x = theta^2 for reference
        ax.plot(
            theta_range,
            theta_range**2,
            color="k",
            # alpha=0.3,
            linestyle="--",
            label=r"$x=\theta^2$",
        )
        return contour

    # --- Plot 1: Neural Likelihood Estimation (NLE) ---
    ax_nle = axes[0, 0]
    setup_axis(ax_nle, r"NLE: Likelihood $P(X | \theta)$", P_x_given_theta)

    # Overlay Prior on NLE with independent axis
    ax_prior = ax_nle.twinx()
    prior_1d = norm.pdf(theta_range, loc=mu_theta, scale=sigma_theta)

    # Plot the prior as a red dashed line
    ax_prior.plot(
        theta_range,
        prior_1d,
        color="red",
        # linestyle="--",
        linewidth=2.0,
        label=r"Prior $P(\theta)$",
    )

    # Style the twin axis explicitly (Red axis on the right)
    # ax_prior.set_ylabel(r"Prior Density $P(\theta)$", color="red", fontsize=12)
    # ax_prior.tick_params(axis="y", labelcolor="red")
    # ax_prior.spines["right"].set_color("red")
    ax_prior.set_yticks([])
    ax_prior.set_ylim(0, np.max(prior_1d) * 1.2)  # Give it a little headroom
    ax_prior.text(
        mu_theta,
        np.max(prior_1d) * 1.05,
        "Prior",
        color="red",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

    # Annotation for Likelihood
    # Show that vertical slices (fixed theta) have constant width
    ax_nle.annotate(
        "", xy=(0, 2.5), xytext=(0, -1.5), arrowprops=dict(arrowstyle="<->", color="white", lw=2)
    )
    ax_nle.text(
        0, 3.0, "Normalized Vertically", color="white", ha="center", fontsize=9, fontweight="bold"
    )

    # --- Plot 2: Neural Posterior Estimation (NPE) ---
    ax_npe = axes[0, 1]
    setup_axis(ax_npe, r"NPE: Posterior $P(\theta | X)$", P_theta_given_x)
    # Annotation for horizontal normalization
    ax_npe.annotate(
        "", xy=(1.5, 1), xytext=(-1.5, 1), arrowprops=dict(arrowstyle="<->", color="white", lw=1.5)
    )
    ax_npe.text(
        0, 1.3, "Normalized Horizontally", color="white", ha="center", fontsize=9, fontweight="bold"
    )

    # --- Plot 3: Neural Joint Estimation (NJE) ---
    ax_nje = axes[1, 0]
    setup_axis(ax_nje, r"NJE: Joint $P(X, \theta)$", P_joint)
    ax_nje.text(
        0, 6.0, "Globally Normalized", color="white", ha="center", fontsize=9, fontweight="bold"
    )

    # --- Plot 4: Neural Ratio Estimation (NRE) ---
    ax_nre = axes[1, 1]
    # The ratio can be very large, so we sometimes plot log-ratio, but raw ratio shows the contrast well here.
    setup_axis(ax_nre, r"NRE: Ratio $\frac{P(X, \theta)}{P(X)P(\theta)}$", P_ratio)
    ax_nre.text(
        0,
        7,
        "High where X and $\\theta$\nare covariant",
        color="white",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

    fig.suptitle(
        f"Simulation-Based Inference Objectives: Parabolic Case ($x = \\theta^2 + \\epsilon,~\\epsilon \\sim \\mathcal{{N}}(0,{sigma_x:.1f}^2)$)",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig("SBIdemo.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_sbi_distributions()
