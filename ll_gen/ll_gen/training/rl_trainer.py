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
        assert self._optimizer is not None
        assert self._disposal_engine is not None

        # 1. Generate proposal (no_grad: avoid building graph during sampling)
        with torch.no_grad():
            proposal = self.generator.generate(prompt)

        # 2. Dispose (deterministic execution + validation)
        result = self._disposal_engine.dispose(proposal, export=False)

        # 3. Compute reward
        reward = compute_reward(
            result,
            config=self.feedback_config,
            target_dimensions=target_dimensions,
        )

        # 4. Update baseline (EMA), seeding with first observed reward
        if self._step_count == 0:
            self._baseline = reward
        else:
            self._baseline = (
                self.baseline_decay * self._baseline +
                (1.0 - self.baseline_decay) * reward
            )

        # 5. Compute advantage
        advantage = reward - self._baseline

        # 6. Policy gradient loss
        # Re-run the model with the generated token sequence to get log_probs
        # This is the key: we need log probabilities for the generated tokens

        loss_value = 0.0
        entropy_value = 0.0

        try:
            # Extract token IDs from proposal if available
            token_ids = None
            if hasattr(proposal, "token_ids"):
                token_ids = proposal.token_ids
            elif hasattr(proposal, "command_sequence"):
                # For command-based proposals, extract or synthesize token IDs
                token_ids = self._extract_token_ids(proposal)

            if token_ids is not None and len(token_ids) > 0:
                # Zero gradients before teacher-forcing forward pass
                self._optimizer.zero_grad()

                # Teacher-forcing: run model with token_ids to get log_probs
                log_probs, entropy_value = self._get_log_probs(token_ids)

                if log_probs is not None:
                    # Policy gradient: -advantage * log_probs - entropy_bonus
                    # (negative because we want to maximize reward)
                    policy_loss = -advantage * log_probs.sum()
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
                _log.warning("Could not extract token_ids from proposal")

        except Exception as e:
            _log.error("Error during policy gradient computation: %s", e)
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

        Args:
            token_ids: Array of token IDs to compute log_probs for.

        Returns:
            Tuple of (log_probs tensor or None, entropy value).
            log_probs shape: (seq_len,) or (seq_len * num_params,)
            entropy: scalar value
        """
        import torch

        if self.generator._model is None:
            return None, 0.0

        try:
            # Convert to tensor
            if isinstance(token_ids, np.ndarray):
                token_ids_t = torch.from_numpy(token_ids).long().to(self.device)
            else:
                token_ids_t = token_ids.to(self.device)

            # Forward pass (no torch.no_grad — gradients must flow for training)
            # The exact interface depends on your model.
            # Assumes model returns logits or log_probs directly
            output = self.generator._model(token_ids_t.unsqueeze(0))

            # Extract log probabilities from output
            # This is model-specific; adapt as needed
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, tuple) and len(output) > 0:
                logits = output[0]
            else:
                logits = output

            # Compute log_probs via softmax
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Gather log-probs for actually chosen tokens only (REINFORCE)
            # logits shape: (1, seq_len, vocab_size) or (seq_len, vocab_size)
            if log_probs.dim() == 3:
                log_probs = log_probs.squeeze(0)  # (seq_len, vocab_size)

            # Shift: teacher forcing means logits[t] predicts token[t+1]
            if len(token_ids_t) > 1:
                shifted_log_probs = log_probs[:-1]   # (seq_len-1, vocab)
                shifted_targets = token_ids_t[1:]     # (seq_len-1,)
                chosen_log_probs = shifted_log_probs.gather(
                    1, shifted_targets.unsqueeze(1)
                ).squeeze(1)
                log_prob_sum = chosen_log_probs.sum()

                # Entropy over valid shifted positions
                shifted_probs = torch.exp(shifted_log_probs)
                entropy = -(shifted_probs * shifted_log_probs).sum(dim=-1).mean()
            else:
                # Degenerate single-token case
                log_prob_sum = log_probs.sum()
                probs = torch.exp(log_probs)
                entropy = -(probs * log_probs).sum()

            return log_prob_sum, float(entropy.detach().cpu())

        except Exception as e:
            _log.warning("Error computing log_probs: %s", e)
            return None, 0.0

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
            np.random.shuffle(dataset)

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
        assert self._disposal_engine is not None

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
        assert self._optimizer is not None

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
        history_path = str(path).replace(".pt", "_history.json")
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
        assert self._optimizer is not None

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
        history_path = str(path).replace(".pt", "_history.json")
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
