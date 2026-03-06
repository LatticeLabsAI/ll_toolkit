"""RL alignment trainer using disposal validation as oracle.

Implements policy gradient training with REINFORCE + baseline subtraction,
using disposal engine feedback and reward signals for RL optimization.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ll_gen.config import DisposalConfig, FeedbackConfig
from ll_gen.disposal.engine import DisposalEngine
from ll_gen.feedback.reward_signal import compute_reward
from ll_gen.generators.base import BaseNeuralGenerator
from ll_gen.training.metrics import MetricsComputer

_log = logging.getLogger(__name__)


class RLAlignmentTrainer:
    """RL alignment trainer using disposal validation as oracle.

    Implements REINFORCE policy gradient with baseline subtraction.
    The generator's neural model is trained to maximize disposal rewards
    (validity, compilation, geometry correctness).

    Attributes:
        generator: Neural generator to train.
        disposal_config: Configuration for validation/repair.
        feedback_config: Configuration for reward signals.
        learning_rate: Optimizer learning rate.
        baseline_decay: EMA decay for reward baseline.
        entropy_coeff: Entropy bonus coefficient for exploration.
        max_grad_norm: Gradient clipping threshold.
        device: Training device ("cpu" or "cuda").
        output_dir: Directory for saving checkpoints.
    """

    def __init__(
        self,
        generator: BaseNeuralGenerator,
        disposal_config: DisposalConfig | None = None,
        feedback_config: FeedbackConfig | None = None,
        learning_rate: float = 1e-5,
        baseline_decay: float = 0.99,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        output_dir: str = "training_output",
        seed: int | None = None,
    ) -> None:
        """Initialize the RL alignment trainer.

        Args:
            generator: The neural generator to train.
            disposal_config: Disposal/validation configuration. Uses defaults if None.
            feedback_config: Reward signal configuration. Uses defaults if None.
            learning_rate: Optimizer learning rate (Adam).
            baseline_decay: EMA decay for running reward baseline.
            entropy_coeff: Weight of entropy bonus for exploration.
            max_grad_norm: Gradient clipping threshold.
            device: Target device ("cpu" or "cuda").
            output_dir: Directory for saving checkpoints and logs.
            seed: Random seed for reproducible shuffling. None for non-deterministic.
        """
        self.generator = generator
        self.disposal_config = disposal_config or DisposalConfig()
        self.feedback_config = feedback_config or FeedbackConfig()
        self.learning_rate = learning_rate
        self.baseline_decay = baseline_decay
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-initialized components
        self._disposal_engine: DisposalEngine | None = None
        self._optimizer: Any | None = None
        self._baseline: float = 0.0
        self._train_history: list[dict[str, Any]] = []
        self._step_count: int = 0
        self._metrics_computer = MetricsComputer()
        self._rng = np.random.default_rng(seed)

    def _init_training(self) -> None:
        """Initialize training components lazily.

        Sets up optimizer, disposal engine, and baseline tracking.
        Called once before first training step.
        """
        if self._optimizer is not None:
            return  # Already initialized

        import torch.optim as optim

        # Verify generator has a model
        if self.generator._model is None:
            raise RuntimeError(
                "Generator model not initialized. "
                "Call generator.load() or ensure model is set."
            )

        # Create optimizer over model parameters
        self._optimizer = optim.Adam(
            self.generator._model.parameters(),
            lr=self.learning_rate,
        )

        # Initialize disposal engine
        self._disposal_engine = DisposalEngine(
            disposal_config=self.disposal_config,
            feedback_config=self.feedback_config,
            output_dir=str(self.output_dir / "disposed"),
        )

        # Initialize baseline
        self._baseline = 0.0

        _log.info(
            "Training initialized: lr=%.2e, baseline_decay=%.3f, entropy=%.3f",
            self.learning_rate,
            self.baseline_decay,
            self.entropy_coeff,
        )

    def train_step(
        self,
        prompt: str,
        target_dimensions: tuple | None = None,
    ) -> dict[str, float]:
        """Execute a single training step with REINFORCE.

        1. Generate proposal from prompt
        2. Dispose (validate + repair)
        3. Compute reward
        4. Update baseline (EMA)
        5. Compute policy gradient loss with advantage
        6. Backward + gradient clipping + optimizer step

        Args:
            prompt: User prompt describing the shape.
            target_dimensions: Optional target bbox dimensions for semantic bonus.

        Returns:
            Dictionary with keys:
            - "reward": scalar reward from disposal
            - "advantage": reward minus baseline
            - "loss": scalar training loss
            - "baseline": updated baseline value
            - "is_valid": whether result passed validation

        Raises:
            RuntimeError: If training not initialized or generation fails.
        """
        import torch

        self._init_training()
        if self._optimizer is None:
            raise RuntimeError("Optimizer not initialized — call _init_training() first")
        if self._disposal_engine is None:
            raise RuntimeError("Disposal engine not initialized — call _init_training() first")

        # 1. Generate proposal WITH gradients (log-probs on the sampled trajectory)
        proposal = self.generator.generate_for_training(prompt)

        # 2. Dispose (deterministic execution + validation — no grad needed)
        result = self._disposal_engine.dispose(proposal, export=False)

        # 3. Compute reward (scalar, non-differentiable)
        reward = compute_reward(
            result,
            config=self.feedback_config,
            target_dimensions=target_dimensions,
        )

        # 4. Compute advantage BEFORE updating baseline so step-0 is not wasted
        advantage = reward - self._baseline

        # 5. Update baseline (EMA), seeding with first observed reward
        if self._step_count == 0:
            self._baseline = reward
        else:
            self._baseline = (
                self.baseline_decay * self._baseline +
                (1.0 - self.baseline_decay) * reward
            )

        # 6. Policy gradient loss using log-probs from the SAME trajectory
        loss_value = 0.0
        step_failed = False
        entropy_value = proposal.entropy if proposal.entropy is not None else 0.0

        try:
            if proposal.log_probs is not None:
                self._optimizer.zero_grad()

                # REINFORCE: -advantage * log_prob(sampled trajectory)
                policy_loss = -advantage * proposal.log_probs
                entropy_bonus = -self.entropy_coeff * entropy_value

                total_loss = policy_loss + entropy_bonus

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.generator._model.parameters(),
                    self.max_grad_norm,
                )

                # Optimizer step
                self._optimizer.step()

                loss_value = float(total_loss.detach().cpu())
            else:
                _log.warning(
                    "Proposal has no log_probs — generator may not support "
                    "generate_for_training(); skipping gradient update"
                )

        except (KeyboardInterrupt, SystemExit):
            raise
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "cuda" in err_msg:
                _log.critical(
                    "Fatal GPU/memory error during backward pass — "
                    "re-raising to prevent corrupt training state: %s", e
                )
                raise
            _log.error(
                "RuntimeError during policy gradient computation "
                "(step %d): %s", self._step_count, e,
                exc_info=True,
            )
            loss_value = 0.0
            step_failed = True
        except (ValueError, TypeError) as e:
            _log.error(
                "Recoverable error during policy gradient computation "
                "(step %d): %s", self._step_count, e,
                exc_info=True,
            )
            loss_value = 0.0
            step_failed = True
        else:
            step_failed = False

        # Track step
        self._step_count += 1

        step_result = {
            "reward": float(reward),
            "advantage": float(advantage),
            "loss": loss_value,
            "baseline": float(self._baseline),
            "is_valid": float(result.is_valid),
            "failed": step_failed,
        }

        # Only record successful steps in training history
        if not step_failed:
            self._train_history.append(step_result)

        return step_result

    def _extract_token_ids(self, proposal: Any) -> np.ndarray | None:
        """Extract token IDs from a proposal.

        Handles CommandSequenceProposal and other proposal types.
        Used by external callers (e.g. evaluation, logging) that need
        token IDs from an already-generated proposal.

        Note:
            For RL training, use ``generator.generate_for_training()``
            instead — it computes log-probs on the same trajectory.

        Args:
            proposal: A BaseProposal subclass.

        Returns:
            Array of token IDs, or None if extraction fails.
        """
        # DeepCAD vocabulary mapping
        COMMAND_TYPE_MAP = {
            "SOL": 0,
            "LINE": 1,
            "ARC": 2,
            "CIRCLE": 3,
            "EXTRUDE": 4,
            "EOS": 5,
        }
        PARAM_OFFSET = 12

        try:
            if hasattr(proposal, "token_ids"):
                return proposal.token_ids
            elif hasattr(proposal, "command_sequence"):
                # For CommandSequenceProposal, synthesize token IDs from commands
                commands = proposal.command_sequence
                token_ids = []
                for cmd in commands:
                    if hasattr(cmd, "token_id"):
                        token_ids.append(cmd.token_id)
                    elif isinstance(cmd, dict):
                        # Map command_type string to DeepCAD vocabulary ID
                        cmd_type = cmd.get("command_type", "")
                        cmd_token = COMMAND_TYPE_MAP.get(
                            cmd_type.upper(), COMMAND_TYPE_MAP.get(cmd_type, 0)
                        )
                        token_ids.append(cmd_token)
                        # Include parameter token IDs offset by PARAM_OFFSET
                        params = cmd.get("parameters", [])
                        param_mask = cmd.get("parameter_mask", None)
                        for j, p in enumerate(params):
                            if param_mask is not None and j < len(param_mask) and not param_mask[j]:
                                continue
                            token_ids.append(int(p) + PARAM_OFFSET)
                    else:
                        # Object with command_type attribute
                        cmd_type = getattr(cmd, "command_type", "")
                        cmd_token = COMMAND_TYPE_MAP.get(
                            str(cmd_type).upper(), 0
                        )
                        token_ids.append(cmd_token)
                return np.array(token_ids, dtype=np.int64)
            else:
                return None
        except Exception as e:
            _log.warning("Failed to extract token_ids: %s", e)
            return None

    def _get_log_probs(
        self,
        token_ids: np.ndarray,
    ) -> tuple[Any | None, float]:
        """Get log probabilities for a token sequence via teacher-forcing.

        .. deprecated::
            This method runs a separate forward pass that is disconnected
            from the sampling trajectory, producing biased gradients for
            stochastic models (VAE, diffusion).  Use
            ``generator.generate_for_training()`` for RL training instead.

        Args:
            token_ids: Array of token IDs to compute log_probs for.

        Returns:
            Tuple of (log_probs tensor or None, entropy value).
            log_probs shape: (seq_len,) or (seq_len * num_params,)
            entropy: scalar value
        """
        raise NotImplementedError(
            "_get_log_probs() ran a forward pass disconnected from the "
            "sampling trajectory, producing biased gradients for stochastic "
            "models (VAE, diffusion). Use generator.generate_for_training() "
            "which returns log_probs on the actual sampled trajectory via "
            "proposal.log_probs."
        )

    def train_epoch(
        self,
        dataset: list[dict[str, Any]],
        batch_size: int = 4,
        shuffle: bool = True,
    ) -> dict[str, float]:
        """Train for one epoch on a dataset.

        Each sample in dataset should have at least a "prompt" key.
        Optionally includes "target_dimensions" for semantic matching.

        Args:
            dataset: List of dicts with "prompt" and optional "target_dimensions".
            batch_size: Batch size (processes one sample at a time; used for reporting).
            shuffle: Whether to shuffle dataset before training.

        Returns:
            Dictionary with aggregated metrics:
            - "mean_reward": average reward across epoch
            - "mean_loss": average loss
            - "validity_rate": fraction of valid results
            - "epoch_time_ms": wall-clock time in milliseconds
        """
        epoch_start = time.time()

        # Copy to avoid mutating the caller's list
        dataset = list(dataset)
        if shuffle:
            self._rng.shuffle(dataset)

        rewards = []
        losses = []
        validities = []

        _log.info("Starting epoch with %d samples", len(dataset))

        # Set model to training mode for correct dropout/batchnorm behavior
        if self.generator._model is not None:
            self.generator._model.train()

        for i, sample in enumerate(dataset):
            prompt = sample.get("prompt", "")
            target_dims = sample.get("target_dimensions", None)

            try:
                step_result = self.train_step(prompt, target_dims)
                rewards.append(step_result["reward"])
                losses.append(step_result["loss"])
                validities.append(step_result["is_valid"])

                if (i + 1) % max(1, len(dataset) // 10) == 0:
                    _log.info(
                        "Epoch progress: %d/%d, reward=%.4f, loss=%.4f",
                        i + 1,
                        len(dataset),
                        step_result["reward"],
                        step_result["loss"],
                    )

            except Exception as e:
                _log.error("Error in train_step for sample %d: %s", i, e)
                continue

        epoch_time = (time.time() - epoch_start) * 1000  # ms

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        mean_loss = float(np.mean(losses)) if losses else 0.0
        validity_rate = float(np.mean(validities)) if validities else 0.0

        result = {
            "mean_reward": mean_reward,
            "mean_loss": mean_loss,
            "validity_rate": validity_rate,
            "epoch_time_ms": epoch_time,
        }

        _log.info("Epoch complete: %s", result)

        return result

    def evaluate(self, test_set: list[dict[str, Any]]) -> dict[str, float]:
        """Evaluate on a test set without gradient updates.

        Args:
            test_set: List of test samples with "prompt" keys.

        Returns:
            Dictionary with metrics:
            - "validity_rate": fraction of valid results
            - "compile_rate": fraction with shape
            - "mean_reward": average reward
            - "reward_std": standard deviation of reward
        """
        import torch

        self._init_training()
        if self._disposal_engine is None:
            raise RuntimeError("Disposal engine not initialized — call _init_training() first")

        # Set model to eval mode for correct dropout/batchnorm behavior
        if self.generator._model is not None:
            self.generator._model.eval()

        results = []

        with torch.no_grad():
            for sample in test_set:
                prompt = sample.get("prompt", "")
                target_dims = sample.get("target_dimensions", None)

                try:
                    proposal = self.generator.generate(prompt)
                    result = self._disposal_engine.dispose(proposal, export=False)
                    reward = compute_reward(
                        result,
                        config=self.feedback_config,
                        target_dimensions=target_dims,
                    )
                    result.reward_signal = reward
                    results.append(result)

                except Exception as e:
                    _log.error("Error evaluating sample: %s", e)
                    continue

        # Compute metrics
        metrics = self._metrics_computer.compute_all(results)

        return {
            "validity_rate": metrics.validity_rate,
            "compile_rate": metrics.compile_rate,
            "mean_reward": metrics.mean_reward,
            "reward_std": metrics.reward_std,
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint (model, optimizer, baseline, history).

        Args:
            path: Path to save checkpoint to.
        """
        import torch

        self._init_training()
        if self._optimizer is None:
            raise RuntimeError("Optimizer not initialized — call _init_training() first")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "step_count": self._step_count,
            "baseline": self._baseline,
        }

        # Save model and optimizer state
        if self.generator._model is not None:
            checkpoint["model_state_dict"] = (
                self.generator._model.state_dict()
            )
            checkpoint["optimizer_state_dict"] = (
                self._optimizer.state_dict()
            )

        torch.save(checkpoint, path)

        # Save train_history as separate JSON (avoids weights_only=True issues)
        history_path = path.with_name(path.stem + "_history.json")
        with open(history_path, "w") as f:
            json.dump(self._train_history, f)

        _log.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        import torch

        self._init_training()
        if self._optimizer is None:
            raise RuntimeError("Optimizer not initialized — call _init_training() first")

        path = Path(path)
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=True,
            )
        except TypeError:
            # PyTorch < 2.0 does not support weights_only
            checkpoint = torch.load(path, map_location=self.device)

        self._step_count = checkpoint.get("step_count", 0)
        self._baseline = checkpoint.get("baseline", 0.0)

        # Load train_history from separate JSON file
        history_path = path.with_name(path.stem + "_history.json")
        try:
            with open(history_path, "r") as f:
                self._train_history = json.load(f)
        except FileNotFoundError:
            _log.warning("No history file found at %s; starting fresh", history_path)
            self._train_history = checkpoint.get("train_history", [])

        # Load model and optimizer state
        if "model_state_dict" in checkpoint and self.generator._model is not None:
            self.generator._model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        if "optimizer_state_dict" in checkpoint:
            self._optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        _log.info("Checkpoint loaded from %s (step %d)", path, self._step_count)

    def get_training_history(self) -> list[dict[str, float]]:
        """Get the training history.

        Returns:
            List of dicts from each training step.
        """
        return self._train_history
