import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set()

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from msp_tsne import ParametricTSNE, MultiscaleParametricTSNE


def plot_side_by_side(X_p_train, X_p_test, X_ms_train, X_ms_test, y_train, y_test):
    """Create a side-by-side plot comparing Parametric t-SNE vs Multi-Scale Parametric t-SNE.
    Both subplots show the full dataset (train + test) embeddings colored by label.
    """
    colors = cm.rainbow(np.linspace(0, 1, 10))

    # Combine train and test for plotting
    Xp_full = np.vstack((X_p_train, X_p_test))
    Xms_full = np.vstack((X_ms_train, X_ms_test))
    y_full = np.concatenate((y_train, y_test))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ParametricTSNE subplot
    axes[0].set_title('Parametric t-SNE')
    for c in range(10):
        axes[0].scatter(
            Xp_full[y_full == c, 0], Xp_full[y_full == c, 1], s=8, color=colors[c], alpha=.6
        )
    axes[0].set_xlabel('dim 1')
    axes[0].set_ylabel('dim 2')

    # MultiscaleParametricTSNE subplot
    axes[1].set_title('Multi-Scale Parametric t-SNE')
    for c in range(10):
        axes[1].scatter(
            Xms_full[y_full == c, 0], Xms_full[y_full == c, 1], s=8, color=colors[c], alpha=.6
        )
    axes[1].set_xlabel('dim 1')
    axes[1].set_ylabel('dim 2')

    fig.tight_layout()
    return fig, axes


if __name__ == '__main__':
    # Load data
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=0
    )

    # Pipeline with ParametricTSNE
    pipe_parametric = Pipeline([
        ('scaler', StandardScaler()),
        ('pt', ParametricTSNE(
            n_components=2,
            perplexity=30.0,
            n_iter=250,
            verbose=1
        ))
    ])

    # Pipeline with MultiscaleParametricTSNE
    pipe_multiscale = Pipeline([
        ('scaler', StandardScaler()),
        ('mst', MultiscaleParametricTSNE(
            n_components=2,
            n_iter=250,
            verbose=1
        ))
    ])

    # Fit on training and transform both train and test for each method
    X_p_train = pipe_parametric.fit_transform(X_train)
    X_p_test = pipe_parametric.transform(X_test)

    X_ms_train = pipe_multiscale.fit_transform(X_train)
    X_ms_test = pipe_multiscale.transform(X_test)

    # Plot comparison
    plot_side_by_side(X_p_train, X_p_test, X_ms_train, X_ms_test, y_train, y_test)

    # Show
    plt.show()


