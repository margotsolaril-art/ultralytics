# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Base callbacks for Ultralytics training, validation, prediction, and export processes."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics.engine.exporter import Exporter
    from ultralytics.engine.predictor import BasePredictor
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.engine.validator import BaseValidator

# Trainer callbacks ----------------------------------------------------------------------------------------------------


def on_pretrain_routine_start(trainer: BaseTrainer) -> None:
    """Called before the pretraining routine starts."""
    pass


def on_pretrain_routine_end(trainer: BaseTrainer) -> None:
    """Called after the pretraining routine ends."""
    pass


def on_train_start(trainer: BaseTrainer) -> None:
    """Called when the training starts."""
    pass


def on_train_epoch_start(trainer: BaseTrainer) -> None:
    """Called at the start of each training epoch."""
    pass


def on_train_batch_start(trainer: BaseTrainer) -> None:
    """Called at the start of each training batch."""
    pass


def optimizer_step(trainer: BaseTrainer) -> None:
    """Called when the optimizer takes a step."""
    pass


def on_before_zero_grad(trainer: BaseTrainer) -> None:
    """Called before the gradients are set to zero."""
    pass


def on_train_batch_end(trainer: BaseTrainer) -> None:
    """Called at the end of each training batch."""
    pass


def on_train_epoch_end(trainer: BaseTrainer) -> None:
    """Called at the end of each training epoch."""
    pass


def on_fit_epoch_end(trainer: BaseTrainer) -> None:
    """Called at the end of each fit epoch (train + val)."""
    pass


def on_model_save(trainer: BaseTrainer) -> None:
    """Called when the model is saved."""
    pass


def on_train_end(trainer: BaseTrainer) -> None:
    """Called when the training ends."""
    pass


def on_params_update(trainer: BaseTrainer) -> None:
    """Called when the model parameters are updated."""
    pass


def teardown(trainer: BaseTrainer) -> None:
    """Called during the teardown of the training process."""
    pass


# Validator callbacks --------------------------------------------------------------------------------------------------


def on_val_start(validator: BaseValidator) -> None:
    """Called when the validation starts."""
    pass


def on_val_batch_start(validator: BaseValidator) -> None:
    """Called at the start of each validation batch."""
    pass


def on_val_batch_end(validator: BaseValidator) -> None:
    """Called at the end of each validation batch."""
    pass


def on_val_end(validator: BaseValidator) -> None:
    """Called when the validation ends."""
    pass


# Predictor callbacks --------------------------------------------------------------------------------------------------


def on_predict_start(predictor: BasePredictor) -> None:
    """Called when the prediction starts."""
    pass


def on_predict_batch_start(predictor: BasePredictor) -> None:
    """Called at the start of each prediction batch."""
    pass


def on_predict_batch_end(predictor: BasePredictor) -> None:
    """Called at the end of each prediction batch."""
    pass


def on_predict_postprocess_end(predictor: BasePredictor) -> None:
    """Called after the post-processing of the prediction ends."""
    pass


def on_predict_end(predictor: BasePredictor) -> None:
    """Called when the prediction ends."""
    pass


# Exporter callbacks ---------------------------------------------------------------------------------------------------


def on_export_start(exporter: Exporter) -> None:
    """Called when the model export starts."""
    pass


def on_export_end(exporter: Exporter) -> None:
    """Called when the model export ends."""
    pass


default_callbacks = {
    # Run in trainer
    "on_pretrain_routine_start": [on_pretrain_routine_start],
    "on_pretrain_routine_end": [on_pretrain_routine_end],
    "on_train_start": [on_train_start],
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_batch_start": [on_train_batch_start],
    "optimizer_step": [optimizer_step],
    "on_before_zero_grad": [on_before_zero_grad],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_fit_epoch_end": [on_fit_epoch_end],  # fit = train + val
    "on_model_save": [on_model_save],
    "on_train_end": [on_train_end],
    "on_params_update": [on_params_update],
    "teardown": [teardown],
    # Run in validator
    "on_val_start": [on_val_start],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_end": [on_val_end],
    # Run in predictor
    "on_predict_start": [on_predict_start],
    "on_predict_batch_start": [on_predict_batch_start],
    "on_predict_postprocess_end": [on_predict_postprocess_end],
    "on_predict_batch_end": [on_predict_batch_end],
    "on_predict_end": [on_predict_end],
    # Run in exporter
    "on_export_start": [on_export_start],
    "on_export_end": [on_export_end],
}


def get_default_callbacks() -> defaultdict[str, list]:
    """Get the default callbacks for Ultralytics training, validation, prediction, and export processes.

    Returns:
        A defaultdict mapping callback event names to lists of callback functions.

    Examples:
        >>> callbacks = get_default_callbacks()
        >>> print(list(callbacks.keys()))  # show all available callback events
        ['on_pretrain_routine_start', 'on_pretrain_routine_end', ...]
    """
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance: BaseTrainer | BasePredictor | BaseValidator | Exporter) -> None:
    """Add integration callbacks to the instance's callbacks dictionary.

    This function loads and adds various integration callbacks to the provided instance. The specific callbacks added
    depend on the type of instance provided. All instances receive HUB and platform callbacks, while Trainer instances
    also receive additional callbacks for various integrations like ClearML, Comet, DVC, MLflow, Neptune, Ray Tune,
    TensorBoard, and Weights & Biases.

    Args:
        instance: The object instance to which callbacks will be added. The type of instance determines which
            callbacks are loaded.

    Examples:
        >>> from ultralytics.engine.trainer import BaseTrainer
        >>> trainer = BaseTrainer()
        >>> add_integration_callbacks(trainer)
    """
    from .hub import callbacks as hub_cb
    from .platform import callbacks as platform_cb

    # Load Ultralytics callbacks
    callbacks_list = [hub_cb, platform_cb]

    # Load training callbacks
    if "Trainer" in instance.__class__.__name__:
        from .clearml import callbacks as clear_cb
        from .comet import callbacks as comet_cb
        from .dvc import callbacks as dvc_cb
        from .mlflow import callbacks as mlflow_cb
        from .neptune import callbacks as neptune_cb
        from .raytune import callbacks as tune_cb
        from .tensorboard import callbacks as tb_cb
        from .wb import callbacks as wb_cb

        callbacks_list.extend([clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb])

    # Add the callbacks to the callbacks dictionary
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)
