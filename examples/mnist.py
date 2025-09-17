import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set()

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from msp_tsne import MultiscaleParametricTSNE

def plot(X_train, X_test, y_train, y_test):
    """Creates and configures the plots without showing them."""
    colors = cm.rainbow(np.linspace(0, 1, 10))
    
    # Training set plot
    fig1, ax1 = plt.subplots()
    ax1.set_title('MNIST - Training Set Embedding')
    for c in range(10):
        ax1.scatter(X_train[y_train == c, 0], X_train[y_train == c, 1], s=8, color=colors[c], alpha=.6)
    fig1.tight_layout()

    # Combined training and test set plot
    fig2, ax2 = plt.subplots()
    ax2.set_title('MNIST - Full Dataset Embedding')
    X_full = np.vstack((X_train, X_test))
    y_full = np.concatenate((y_train, y_test))
    for c in range(10):
        ax2.scatter(X_full[y_full == c, 0], X_full[y_full == c, 1], s=8, color=colors[c], alpha=.6)
    fig2.tight_layout()

if __name__ == '__main__':
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

    # Corrected Pipeline definition (extra parentheses removed)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('msp_tsne', MultiscaleParametricTSNE(n_components=2,
                                            n_iter=250,
                                            verbose=1))
    ])

    # Use fit_transform for efficiency on the training data.
    # This fits the pipeline and transforms the training data in one step.
    X_tr_2d = pipe.fit_transform(X_train)

    # Transform the test data using the already fitted pipeline.
    X_ts_2d = pipe.transform(X_test)

    # Generate the plots.
    plot(X_tr_2d, X_ts_2d, y_train, y_test)

    # Display all created figures at the end.
    plt.show()

