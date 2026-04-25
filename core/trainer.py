"""
trainer.py
==========
Unified training engine supporting PyTorch, scikit-learn, and JAX backends.
Provides a consistent fit/predict/evaluate API regardless of backend.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for a training run."""
    model_name: str
    backend: str                        # "pytorch" | "sklearn" | "jax"
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    random_seed: int = 42
    device: str = "cpu"                 # "cpu" | "cuda" | "mps"
    num_workers: int = 0
    verbose: bool = True


@dataclass
class TrainingResult:
    """Outputs from a completed training run."""
    model_name: str
    backend: str
    train_losses: List[float]
    val_losses: List[float]
    best_val_loss: float
    best_epoch: int
    training_time_s: float
    peak_memory_mb: float
    config: TrainingConfig
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def epochs_trained(self) -> int:
        return len(self.train_losses)

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "backend": self.backend,
            "best_val_loss": round(self.best_val_loss, 6),
            "best_epoch": self.best_epoch,
            "epochs_trained": self.epochs_trained,
            "training_time_s": round(self.training_time_s, 3),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }


class BaseTrainer(ABC):
    """Abstract base class for all backend trainers."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._model = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainingResult:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def _track_memory(self) -> float:
        """Return current process memory in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e6
        except ImportError:
            return 0.0


class SklearnTrainer(BaseTrainer):
    """
    Wraps any scikit-learn compatible estimator.

    Supports classifiers and regressors that implement fit/predict.
    Optionally tracks training curve via warm_start or partial_fit.

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> cfg = TrainingConfig(model_name="random_forest", backend="sklearn",
    ...                      hyperparams={"n_estimators": 200, "max_depth": 10})
    >>> trainer = SklearnTrainer(cfg, model_cls=RandomForestClassifier)
    >>> result = trainer.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(self, config: TrainingConfig, model_cls=None, model_instance=None):
        super().__init__(config)
        if model_instance is not None:
            self._model = model_instance
        elif model_cls is not None:
            self._model = model_cls(**config.hyperparams)
        else:
            raise ValueError("Provide either model_cls or model_instance")

    def fit(self, X_train, y_train, X_val, y_val) -> TrainingResult:
        from sklearn.metrics import log_loss

        np.random.seed(self.config.random_seed)
        mem_before = self._track_memory()
        t0 = time.perf_counter()

        self._model.fit(X_train, y_train)

        elapsed = time.perf_counter() - t0
        mem_after = self._track_memory()

        # Compute a single-step "loss" for compatibility
        try:
            train_proba = self._model.predict_proba(X_train)
            val_proba = self._model.predict_proba(X_val)
            train_loss = log_loss(y_train, train_proba)
            val_loss = log_loss(y_val, val_proba)
        except AttributeError:
            # Regressor or model without predict_proba
            from sklearn.metrics import mean_squared_error
            train_loss = mean_squared_error(y_train, self._model.predict(X_train))
            val_loss = mean_squared_error(y_val, self._model.predict(X_val))

        if self.config.verbose:
            logger.info(
                f"[sklearn/{self.config.model_name}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"time={elapsed:.2f}s"
            )

        return TrainingResult(
            model_name=self.config.model_name,
            backend="sklearn",
            train_losses=[train_loss],
            val_losses=[val_loss],
            best_val_loss=val_loss,
            best_epoch=0,
            training_time_s=elapsed,
            peak_memory_mb=max(mem_after - mem_before, 0),
            config=self.config,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        raise AttributeError(f"{type(self._model).__name__} does not support predict_proba")


class PyTorchTrainer(BaseTrainer):
    """
    Trains PyTorch nn.Module models with a standard training loop.

    Features:
    - Automatic mixed precision (AMP)
    - Early stopping
    - LR scheduling
    - GPU/MPS support

    Example
    -------
    >>> import torch.nn as nn
    >>> net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
    >>> cfg = TrainingConfig(model_name="mlp", backend="pytorch", epochs=30,
    ...                      learning_rate=1e-3)
    >>> trainer = PyTorchTrainer(cfg, model=net)
    >>> result = trainer.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(self, config: TrainingConfig, model):
        super().__init__(config)
        self._model = model

    def fit(self, X_train, y_train, X_val, y_val) -> TrainingResult:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.config.random_seed)
        device = torch.device(self.config.device)
        model = self._model.to(device)

        # Build DataLoaders
        def _to_loader(X, y, shuffle):
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)
            ds = TensorDataset(Xt, yt)
            return DataLoader(ds, batch_size=self.config.batch_size, shuffle=shuffle,
                              num_workers=self.config.num_workers, pin_memory=device.type == "cuda")

        train_loader = _to_loader(X_train, y_train, shuffle=True)
        val_loader = _to_loader(X_val, y_val, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        mem_before = self._track_memory()
        t0 = time.perf_counter()

        for epoch in range(self.config.epochs):
            # --- Train ---
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item() * len(xb)
            train_loss = epoch_loss / len(train_loader.dataset)

            # --- Validate ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item() * len(xb)
            val_loss /= len(val_loader.dataset)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best weights
                import copy
                best_weights = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if self.config.verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"[PyTorch/{self.config.model_name}] "
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"train={train_loss:.4f} val={val_loss:.4f}"
                )

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best weights
        model.load_state_dict(best_weights)
        self._model = model
        elapsed = time.perf_counter() - t0
        peak_mem = max(self._track_memory() - mem_before, 0)

        return TrainingResult(
            model_name=self.config.model_name,
            backend="pytorch",
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            training_time_s=elapsed,
            peak_memory_mb=peak_mem,
            config=self.config,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.tensor(X, dtype=torch.float32))
        return F.softmax(logits, dim=1).numpy()


class JAXTrainer(BaseTrainer):
    """
    Trains JAX/Flax models using Optax optimizers.

    Uses functional-style training with JIT-compiled update steps
    for maximum performance.

    Example
    -------
    >>> cfg = TrainingConfig(model_name="jax_mlp", backend="jax", epochs=30)
    >>> trainer = JAXTrainer(cfg, model_def=MyFlaxModel, n_classes=3, n_features=20)
    >>> result = trainer.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(self, config: TrainingConfig, model_def, n_classes: int, n_features: int):
        super().__init__(config)
        self._model_def = model_def
        self._n_classes = n_classes
        self._n_features = n_features
        self._params = None

    def fit(self, X_train, y_train, X_val, y_val) -> TrainingResult:
        try:
            import jax
            import jax.numpy as jnp
            import optax
        except ImportError:
            raise ImportError("JAX and Optax are required for the JAX backend. "
                              "Install with: pip install jax jaxlib optax")

        rng = jax.random.PRNGKey(self.config.random_seed)

        # Initialize model
        model = self._model_def(features=self._n_classes)
        dummy = jnp.ones((1, self._n_features))
        params = model.init(rng, dummy)

        # Optimizer
        optimizer = optax.adamw(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        opt_state = optimizer.init(params)

        # Loss function
        def cross_entropy_loss(params, X, y):
            logits = model.apply(params, X)
            one_hot = jax.nn.one_hot(y, self._n_classes)
            return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

        @jax.jit
        def train_step(params, opt_state, X_batch, y_batch):
            loss, grads = jax.value_and_grad(cross_entropy_loss)(params, X_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        @jax.jit
        def eval_loss(params, X, y):
            return cross_entropy_loss(params, X, y)

        X_tr = jnp.array(X_train)
        y_tr = jnp.array(y_train)
        X_v = jnp.array(X_val)
        y_v = jnp.array(y_val)

        n = len(X_train)
        bs = self.config.batch_size
        n_batches = max(1, n // bs)

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_params = params

        t0 = time.perf_counter()
        mem_before = self._track_memory()

        for epoch in range(self.config.epochs):
            # Shuffle
            perm = jax.random.permutation(jax.random.PRNGKey(epoch), n)
            X_shuf, y_shuf = X_tr[perm], y_tr[perm]

            epoch_loss = 0.0
            for i in range(n_batches):
                xb = X_shuf[i * bs:(i + 1) * bs]
                yb = y_shuf[i * bs:(i + 1) * bs]
                params, opt_state, loss = train_step(params, opt_state, xb, yb)
                epoch_loss += float(loss)

            train_loss = epoch_loss / n_batches
            val_loss = float(eval_loss(params, X_v, y_v))

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_params = params
            else:
                patience_counter += 1

            if self.config.verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"[JAX/{self.config.model_name}] "
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"train={train_loss:.4f} val={val_loss:.4f}"
                )

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self._params = best_params
        self._model = model
        elapsed = time.perf_counter() - t0

        return TrainingResult(
            model_name=self.config.model_name,
            backend="jax",
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            training_time_s=elapsed,
            peak_memory_mb=max(self._track_memory() - mem_before, 0),
            config=self.config,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp
        logits = self._model.apply(self._params, jnp.array(X))
        return np.array(logits.argmax(axis=-1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import jax
        import jax.numpy as jnp
        logits = self._model.apply(self._params, jnp.array(X))
        return np.array(jax.nn.softmax(logits, axis=-1))


def build_trainer(config: TrainingConfig, **kwargs) -> BaseTrainer:
    """
    Factory function to construct the appropriate trainer from a config.

    Parameters
    ----------
    config : TrainingConfig
    **kwargs : Additional arguments passed to the trainer constructor.
               For sklearn: model_cls or model_instance
               For pytorch: model (nn.Module)
               For jax: model_def, n_classes, n_features

    Returns
    -------
    BaseTrainer
    """
    if config.backend == "sklearn":
        return SklearnTrainer(config, **kwargs)
    elif config.backend == "pytorch":
        return PyTorchTrainer(config, **kwargs)
    elif config.backend == "jax":
        return JAXTrainer(config, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {config.backend}. Choose from: sklearn, pytorch, jax")
