import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_sbi_distributions():
    # --- 1. Setup Parameters ---
    # Prior: Theta ~ N(0, 2)
    # We use a wider prior so it doesn't suppress the "arms" of the parabola too much
    # allowing us to see the likelihood shape effects clearly.
    mu_theta = 0.0
    sigma_theta = 1.5

    # Model: Parabolic likelihood
    # x = theta^2 + noise
    # Likelihood: P(x|theta) = N(theta^2, sigma_x)
    sigma_x = 2.0

    # --- 2. Create Grid ---
    # Theta range [-3, 3] covers the prior well.
    # X range [-2, 10] covers the parabola x=theta^2 (up to 9).
    theta_range = np.linspace(-5, 5, 300)
    x_range = np.linspace(-2, 15, 300)
    Theta, X = np.meshgrid(theta_range, x_range)

    # --- 3. Calculate Densities ---

    # A. Prior P(theta)
    P_theta = norm.pdf(Theta, loc=mu_theta, scale=sigma_theta)

    # B. Likelihood P(x | theta)
    # The mean of x is determined by theta^2
    P_x_given_theta = norm.pdf(X, loc=(Theta**2), scale=sigma_x)

    # C. Joint P(x, theta) = P(x | theta) * P(theta)
    P_joint = P_x_given_theta * P_theta

    # D. Marginal P(x) - Numerical Integration
    # Since the model is non-linear, we integrate the joint distribution over theta.
    # We use the trapezoidal rule along axis 1 (the theta axis).
    # Reshape to (N_x, 1) so we can divide the (N_x, N_theta) joint matrix.
    P_x_marginal = np.trapz(P_joint, theta_range, axis=1).reshape(-1, 1)

    # E. Posterior P(theta | x) = P(x, theta) / P(x)
    # Add a tiny epsilon to avoid division by zero in regions where P(x) is negligible
    P_theta_given_x = P_joint / (P_x_marginal + 1e-12)

    # --- 4. Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    def setup_axis(ax, title, z_data):
        # We allow the contour plotter to auto-scale levels for each plot independently
        # This ensures we see the full structure of each distribution
        contour = ax.contourf(Theta, X, z_data, levels=50, cmap="viridis")
        contour.set_rasterized(True)
        ax.set_xlabel(r"Parameter $\theta$")
        ax.set_ylabel(r"Observation $x$")
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_ylim(x_range[0], x_range[-1])
        # Draw the theoretical parabola x = theta^2 for reference
        ax.plot(
            theta_range,
            theta_range**2,
            color="k",
            linewidth=0.5,
            linestyle="--",
            label=r"$x=\theta^2$",
        )

        # Add colorbar
        # cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.1)
        # cbar.ax.tick_params(labelsize=10)
        # cbar.solids.set_rasterized(True)

        return contour

    # Plot 1: Joint Distribution
    setup_axis(
        axes[0],
        r"Joint Distribution $P(X, \theta)$" + "\n" + r"(Target of Neural Ratio Estimation, NRE)",
        P_joint,
    )
    axes[0].text(
        0, 6.0, "Globally Normalized", color="white", ha="center", fontsize=9, fontweight="bold"
    )

    # Plot 2: Likelihood
    setup_axis(
        axes[1],
        r"Likelihood $P(X | \theta)$" + "\n" + r"(Target of Neural Likelihood Estimation, NLE)",
        P_x_given_theta,
    )
    # Annotation for Likelihood
    # Show that vertical slices (fixed theta) have constant width
    axes[1].annotate(
        "", xy=(0, 4), xytext=(0, -1.5), arrowprops=dict(arrowstyle="<->", color="white", lw=2)
    )
    axes[1].text(
        0, 5.0, "Normalized Vertically", color="white", ha="center", fontsize=9, fontweight="bold"
    )

    # --- ADDED: Overlay Prior on Likelihood Plot ---
    # We use a twin axis because the Prior is density over Theta (different units than X)
    ax_prior = axes[1].twinx()
    prior_1d = norm.pdf(theta_range, loc=mu_theta, scale=sigma_theta)
    # Plot the prior as a red dashed line
    ax_prior.plot(theta_range, prior_1d, color="red", linewidth=2.0, label=r"Prior $P(\theta)$")
    # Style the twin axis
    ax_prior.set_ylabel(r"Prior Density $P(\theta)$", color="red", fontsize=12)
    ax_prior.tick_params(axis="y", labelcolor="red")
    ax_prior.spines["right"].set_color("red")
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

    # axes[1].plot(theta_range, P_theta, linewidth = 2, color = "white")

    # Plot 3: Posterior
    setup_axis(
        axes[2],
        r"Posterior $P(\theta | X)$" + "\n" + r"(Target of Neural Posterior Estimation, NPE)",
        P_theta_given_x,
    )
    # Annotation for Posterior
    # Show how the distribution shape changes based on the slope of the parabola

    # Near vertex: Flat slope -> Wide uncertainty -> Low density
    axes[2].annotate(
        "", xy=(1.5, 1), xytext=(-1.5, 1), arrowprops=dict(arrowstyle="<->", color="white", lw=1.5)
    )
    axes[2].text(
        0, 1.3, "Normalized Horizontally", color="white", ha="center", fontsize=9, fontweight="bold"
    )

    # On arms: Steep slope -> Narrow uncertainty -> High density
    # axes[2].annotate("", xy=(2.5, 6), xytext=(1.8, 6),
    #                 arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
    # axes[2].text(2.2, 6.5, "Narrow &\nSharp", color="white", ha="center", fontsize=9, fontweight='bold')

    fig.suptitle(
        f"Visualizing SBI Objectives: Parabolic Case ($x = \\theta^2 + \\epsilon,~\\epsilon \\sim \\mathcal{{N}}(0,{sigma_x:.1f}^2)$)",
        fontsize=18,
        fontweight="bold",
    )
    plt.savefig("SBIdemo.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_sbi_distributions()
