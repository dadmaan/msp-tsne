import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm
import numpy as np
import numba

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


@numba.jit(nopython=True, error_model='numpy')          # https://github.com/numba/numba/issues/4360
def Hbeta(D, beta):

    P = np.exp(-D * beta)
    sumP = np.sum(P)                                    # ACHTUNG! This could be zero!
    H = np.log(sumP) + beta * np.sum(D * P) / sumP      # ACHTUNG! Divide-by-zero possible here!
    P = P / sumP                                        # ACHTUNG! Divide-by-zero possible here!
    
    return H, P

@numba.jit(nopython=True)
def x2p_job(data, max_iteration=50, tol=1e-5):
    i, Di, logU = data
    
    beta = 1.0
    beta_min = -np.inf
    beta_max = np.inf
    
    H, thisP = Hbeta(Di, beta)
    Hdiff = H - logU

    tries = 0
    while tries < max_iteration and np.abs(Hdiff) > tol:
    
        # If not, increase or decrease precision
        if Hdiff > 0:
            beta_min = beta
            if np.isinf(beta_max):      # Numba compatibility: isposinf --> isinf
                beta *= 2.
            else:
                beta = (beta + beta_max) / 2.
        else:
            beta_max = beta
            if np.isinf(beta_min):      # Numba compatibility: isneginf --> isinf
                beta /= 2. 
            else:
                beta = (beta + beta_min) / 2.

        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        tries += 1

    return i, thisP

def x2p(X, perplexity, n_jobs=None):

    n = X.shape[0]
    logU = np.log(perplexity)

    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X + (sum_X.reshape((-1, 1)) - 2 * np.dot(X, X.T))

    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape((n, -1))

    P = np.zeros([n, n])
    for i in range(n):
        P[i, idx[i]] = x2p_job((i,D[i],logU))[1]

    return P

def _noop(*args, **kwargs):
    return None


class ParametricTSNE(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=2, perplexity=30.,
                n_iter=1000,
                batch_size=500,
                early_exaggeration_epochs = 50,
                early_exaggeration_value = 4.,
                early_stopping_epochs = np.inf,
                early_stopping_min_improvement = 1e-2,
                alpha = 1,
                nl1 = 1000,
                nl2 = 500,
                nl3 = 250,
                logdir=None, verbose=0,
                device='auto', lr=1e-3):
        
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.verbose = verbose

        # FFNet architecture
        self.nl1 = nl1
        self.nl2 = nl2
        self.nl3 = nl3

        # Early-exaggeration
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.early_exaggeration_value = early_exaggeration_value
        # Early-stopping
        self.early_stopping_epochs = early_stopping_epochs
        self.early_stopping_min_improvement = early_stopping_min_improvement

        # t-Student params
        self.alpha = alpha

        # Tensorboard
        self.logdir = logdir

        # Optimizer params
        self.lr = lr

        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Internals
        self._model = None
        self._optimizer = None
        self._writer = None
        
    def fit(self, X, y=None):
                
        """fit the model with X"""
        
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        else:
            # HACK! REDUCE 'X' TO MAKE IT MULTIPLE OF BATCH_SIZE!
            m = X.shape[0] % self.batch_size
            if m > 0:
                X = X[:-m]

        n_sample, n_feature = X.shape

        self._log('Building model..', end=' ')
        self._build_model(n_feature, self.n_components)
        self._log('Done')

        self._log('Start training..')
        
        # TensorBoard (PyTorch SummaryWriter)
        if self.logdir is not None:
            self._writer = SummaryWriter(log_dir=self.logdir)
        else:
            self._writer = None

        # Early stopping
        es_patience = self.early_stopping_epochs
        es_loss = np.inf
        es_stop = False

        # Precompute P (once for all!)
        P = self._calculate_P(X)                
        
        epoch = 0
        self._model.train()
        while epoch < self.n_iter and not es_stop:

            # Make copy
            _P = P.copy()

            ## Shuffle entries
            # p_idxs = np.random.permutation(self.batch_size)
    
            # Early exaggeration        
            if epoch < self.early_exaggeration_epochs:
                _P *= self.early_exaggeration_value

            # Actual training
            loss_value = 0.0
            n_batches = 0
            for i in range(0, n_sample, self.batch_size):
                
                batch_slice = slice(i, i + self.batch_size)
                X_batch_np, _P_batch_np = X[batch_slice], _P[batch_slice]
                
                # Shuffle entries
                p_idxs = np.random.permutation(self.batch_size)
                # Shuffle data
                X_batch_np = X_batch_np[p_idxs]
                # Shuffle rows and cols of P
                _P_batch_np = _P_batch_np[p_idxs, :]
                _P_batch_np = _P_batch_np[:, p_idxs]

                # Convert to torch tensors on device
                X_batch = torch.from_numpy(X_batch_np).to(self.device, dtype=torch.float32)
                P_batch = torch.from_numpy(_P_batch_np).to(self.device, dtype=torch.float32)

                # Forward + backward + step
                self._optimizer.zero_grad(set_to_none=True)
                Y = self._model(X_batch)
                loss = self._kl_divergence(P_batch, Y)
                loss.backward()
                self._optimizer.step()

                loss_value += float(loss.detach().cpu().item())
                n_batches += 1
            
            # End-of-epoch: summarize
            loss_value /= n_batches

            if epoch % 10 == 0:
                self._log('Epoch: {0} - Loss: {1:.3f}'.format(epoch, loss_value))
            
            if self._writer is not None:
                self._writer.add_scalar('loss', loss_value, global_step=epoch)
                self._writer.flush()

            # Check early-stopping condition
            if loss_value < es_loss and np.abs(loss_value - es_loss) > self.early_stopping_min_improvement:
                es_loss = loss_value
                es_patience = self.early_stopping_epochs
            else:
                es_patience -= 1

            if es_patience == 0:
                self._log('Early stopping!')
                es_stop = True

            # Going to the next iteration...
            del _P    
            epoch += 1

        if self._writer is not None:
            self._writer.close()

        self._log('Done')

        return self  # scikit-learn does so..
        
    def transform(self, X):

        
        """apply dimensionality reduction to X"""
        # fit should have been called before
        if self.model is None:
            raise sklearn.exceptions.NotFittedError(
                'This ParametricTSNE instance is not fitted yet. Call \'fit\''
                ' with appropriate arguments before using this method.')

        self._log('Predicting embedding points..', end=' ')
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(self.device, dtype=torch.float32)
            Y = self._model(X_tensor)
            X_new = Y.detach().cpu().numpy()

        self._log('Done')

        return X_new

    def fit_transform(self, X, y=None):
        """fit the model with X and apply the dimensionality reduction on X."""
        self.fit(X, y)

        X_new = self.transform(X)
        return X_new

    # ================================ Internals ================================

    def _calculate_P(self, X):
        n = X.shape[0]
        P = np.zeros([n, self.batch_size])
        self._log("Computing P...")
        for i in tqdm(np.arange(0, n, self.batch_size)):
            P_batch = x2p(X[i:i + self.batch_size], self.perplexity)
            P_batch[np.isnan(P_batch)] = 0
            P_batch = P_batch + P_batch.T
            P_batch = P_batch / P_batch.sum()
            P_batch = np.maximum(P_batch, 1e-12)
            P[i:i + self.batch_size] = P_batch
        return P

    def _kl_divergence(self, P, Y):
        # y_true: P (pairwise probabilities); y_pred: Y (low-dim outputs)
        dtype = Y.dtype
        eps = torch.tensor(1e-15, dtype=dtype, device=Y.device)

        # Pairwise distances
        sum_Y = torch.sum(Y * Y, dim=1, keepdim=True)  # (B,1)
        D = sum_Y + sum_Y.T - 2.0 * (Y @ Y.T)  # (B,B)

        Q = torch.pow(1.0 + D / float(self.alpha), - (self.alpha + 1.0) / 2.0)

        # Zero out diagonal and normalize
        batch_size = Y.shape[0]
        mask = 1.0 - torch.eye(batch_size, dtype=dtype, device=Y.device)
        Q = Q * mask
        Q_sum = torch.sum(Q)
        Q = torch.where(Q_sum > 0, Q / Q_sum, Q)
        Q = torch.maximum(Q, eps)

        C = torch.log((P + eps) / (Q + eps))
        C = torch.sum(P * C)
        return C

    def _build_model(self, n_input, n_output):
        layers = []
        layers.append(nn.Linear(n_input, self.nl1))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.nl1, self.nl2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.nl2, self.nl3))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.nl3, n_output))
        self._model = nn.Sequential(*layers).to(self.device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr)

    def _log(self, *args, **kwargs):
        """logging with given arguments and keyword arguments"""
        if self.verbose >= 1:
            print(*args, **kwargs)

    @property
    def model(self):
        return self._model
