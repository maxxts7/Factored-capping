"""
Cross-axis capping experiment library.

This is the core engine behind the cross-capping experiment. It does NOT run
anything on its own -- the orchestrator (run_crosscap.py) calls into the
classes and functions defined here. Think of this file as the toolbox;
run_crosscap.py is the person using the tools.

=== Where this fits in the pipeline ===

    run_crosscap.py (orchestrator)
        │
        ├── WARMUP PHASE
        │     Uses: SteeringExperiment        -- loads the model + assistant axis
        │     Uses: compute_pca_compliance_axis / compute_mean_diff_compliance_axis
        │                                      -- builds the second (correction) axis
        │     Uses: compute_discriminative_thresholds
        │                                      -- finds per-layer firing thresholds
        │
        └── GENERATION PHASE (per prompt)
              Uses: generate_baseline          -- uncapped output (control)
              Uses: generate_capped            -- single-axis capping (baseline method)
              Uses: generate_cross_capped      -- two-axis capping (the new idea)

=== What the key pieces do ===

  SteeringExperiment
      Loads the transformer model and the pre-computed "assistant axis" vector.
      Provides helpers to tokenize prompts and to run a forward pass while
      recording hidden-state activations at every layer.

  _AxisProjectionTracker
      A read-only hook you attach to a layer. During generation it records
      how strongly the hidden state points along a given axis at each step.
      Used for monitoring -- it never changes the model's behavior.

  _CappingHook  (single-axis capping)
      The standard safety intervention. At each generation step it checks
      whether the hidden state's projection onto the assistant axis falls
      below a threshold. If it does, the hook nudges the hidden state back
      up to the threshold along that same axis.

  _CrossAxisCappingHook  (cross-axis capping -- the new idea)
      Like _CappingHook, but splits the job in two:
        - DETECT on the assistant axis (is the model about to comply with
          a jailbreak?)
        - CORRECT on a separate compliance axis (push the hidden state
          toward refusal along a direction learned from PCA)
      This separation is the core hypothesis of the experiment.

  compute_discriminative_thresholds
      Runs a batch of benign and jailbreak prompts through the model,
      measures where their projection distributions land on each axis,
      and picks a threshold halfway between the two means. This is the
      "decision boundary" that tells the capping hooks when to fire.

  compute_pca_compliance_axis / compute_mean_diff_compliance_axis
      Two ways to build the correction axis for cross-axis capping:
        - PCA: pool activations from "refusing" and "compliant" runs,
          center them, and take the first principal component.
        - Mean-diff: simply subtract the mean compliant activation from
          the mean refusing activation and normalise.

  generate_baseline / generate_capped / generate_cross_capped
      The three generation modes. Each one runs the model's .generate()
      method, optionally with capping hooks installed, and returns the
      generated text plus diagnostic data (projection traces, which
      layers fired, how many interventions occurred).

=== The capping formula ===

  Single-axis:   h ← h − v · min(⟨h, v⟩ − τ, 0)
  Cross-axis:    if ⟨h, v_detect⟩ < τ_detect:
                     h ← h − v_correct · min(⟨h, v_correct⟩ − τ_correct, 0)

  In plain English: if the hidden state is "too far" in the jailbreak
  direction (projection below threshold), add just enough of the axis
  vector to bring it back to the threshold. The cross-axis variant only
  fires the correction when the *detection* axis says there's a problem,
  but corrects along a *different* direction.
"""

import logging
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm                       # progress bars
from contextlib import ExitStack            # manages multiple context-manager hooks at once
from typing import Optional
from huggingface_hub import hf_hub_download # download pre-computed axis files from HF
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Logger -- shared across the whole library so all output is consistent
# ---------------------------------------------------------------------------

logger = logging.getLogger("crosscap")
if not logger.handlers:                     # only configure once, even if imported twice
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Model & axis configuration
# ---------------------------------------------------------------------------
#
# Each supported model needs to know:
#   - where to find its pre-computed assistant axis file on HuggingFace
#   - which layer that axis was extracted from (target_layer)
#   - total number of transformer layers (for bounds checking)
#
# The assistant axis is a unit vector in hidden-state space that points in
# the direction the model's activations move when it "acts like an assistant"
# (i.e. answers helpfully rather than refusing). Capping projects activations
# onto this vector and enforces a minimum value.
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "google/gemma-2-27b-it": {
        "hf_axis_file": "gemma-2-27b/assistant_axis.pt",
        "hf_capping_config": None,                        # no capping config available
        "capping_experiment": None,
        "target_layer": 22,
        "num_layers": 46,
    },
    "Qwen/Qwen3-32B": {
        "hf_axis_file": "qwen-3-32b/assistant_axis.pt",
        "hf_capping_config": "qwen-3-32b/capping_config.pt",
        "capping_experiment": "layers_46:54-p0.25",
        "target_layer": 32,
        "num_layers": 64,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "hf_axis_file": "llama-3.3-70b/assistant_axis.pt",
        "hf_capping_config": "llama-3.3-70b/capping_config.pt",
        "capping_experiment": "layers_56:72-p0.25",
        "target_layer": 40,
        "num_layers": 80,
    },
}

# HuggingFace dataset repo that hosts pre-computed axis .pt files
HF_AXIS_REPO = "lu-christina/assistant-axis-vectors"


def load_axis(path: str) -> torch.Tensor:
    """Load a pre-computed assistant axis from a .pt file on disk.

    The file can be either a raw tensor or a dict with an 'axis' key --
    this function handles both formats. Returns shape (num_layers, hidden_dim).
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):              # some files wrap the tensor in a dict
        if "axis" in data:
            return data["axis"]
        raise ValueError("Expected 'axis' key in saved dict")
    return data                             # raw tensor


def download_axis(model_name: str, cache_dir: str = "results") -> str:
    """Download the pre-computed assistant axis for a model from HuggingFace.

    Returns the local file path so load_axis() can read it.
    """
    cfg = MODEL_CONFIGS[model_name]
    return hf_hub_download(
        repo_id=HF_AXIS_REPO,
        filename=cfg["hf_axis_file"],
        repo_type="dataset",
        local_dir=cache_dir,
    )


def load_original_capping(
    model_name: str,
    cache_dir: str = "results",
) -> tuple[dict[int, torch.Tensor], dict[int, float], list[int]]:
    """Load the original paper's exact capping vectors and thresholds.

    Downloads the capping_config.pt from HuggingFace and extracts the
    recommended experiment's per-layer axes and thresholds. The config
    stores vectors in the negated direction (toward role-playing) with
    negative thresholds; this function converts them back to the raw
    assistant-axis direction (toward assistant) with positive thresholds
    so our floor-based _CappingHook produces identical results.

    Returns:
        per_layer_axes:       dict mapping layer_idx -> unit vector (raw direction)
        per_layer_thresholds: dict mapping layer_idx -> threshold (positive)
        cap_layers:           sorted list of layer indices the experiment covers
    """
    cfg = MODEL_CONFIGS[model_name]
    if cfg["hf_capping_config"] is None:
        raise ValueError(
            f"No capping config available for {model_name}. "
            f"Only Qwen/Qwen3-32B and meta-llama/Llama-3.3-70B-Instruct have configs."
        )

    config_path = hf_hub_download(
        repo_id=HF_AXIS_REPO,
        filename=cfg["hf_capping_config"],
        repo_type="dataset",
        local_dir=cache_dir,
    )
    config = torch.load(config_path, map_location="cpu", weights_only=False)

    # Find the recommended experiment
    experiment_id = cfg["capping_experiment"]
    experiment = None
    for exp in config["experiments"]:
        if exp["id"] == experiment_id:
            experiment = exp
            break
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_id}' not found in capping config")

    per_layer_axes = {}
    per_layer_thresholds = {}
    cap_layers = []

    for intervention in experiment["interventions"]:
        if "cap" not in intervention:
            continue
        vec_data = config["vectors"][intervention["vector"]]
        layer_idx = vec_data["layer"]
        # Negate: config stores vectors pointing toward role-playing,
        # we need them pointing toward assistant (raw direction)
        axis = -vec_data["vector"].float()
        axis = axis / axis.norm()
        # Negate threshold: config uses negative values on the negated axis,
        # equivalent to positive values on the raw axis
        threshold = -float(intervention["cap"])

        per_layer_axes[layer_idx] = axis
        per_layer_thresholds[layer_idx] = threshold
        cap_layers.append(layer_idx)

    cap_layers.sort()
    logger.info(
        "Loaded original capping config: experiment=%s, layers=L%d-L%d (%d layers)",
        experiment_id, cap_layers[0], cap_layers[-1], len(cap_layers),
    )
    return per_layer_axes, per_layer_thresholds, cap_layers


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------
#
# Different model families store their transformer layers in different places
# in the object hierarchy (e.g. model.model.layers vs model.transformer.h).
# We try each known path until one works, so the rest of the code doesn't
# need to care which architecture it's running on.
# ---------------------------------------------------------------------------

_LAYER_PATHS = [
    lambda m: m.model.layers,            # Llama, Gemma 2, Qwen, Mistral
    lambda m: m.language_model.layers,    # Gemma 3, LLaVA (vision-language)
    lambda m: m.transformer.h,            # GPT-2/Neo, Bloom
    lambda m: m.transformer.layers,       # Some transformer variants
    lambda m: m.gpt_neox.layers,          # GPT-NeoX
]


def _get_layers(model: nn.Module) -> nn.ModuleList:
    """Try each known attribute path and return the first one that has layers."""
    for path_fn in _LAYER_PATHS:
        try:
            layers = path_fn(model)
            if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except AttributeError:
            continue                      # not this architecture -- try the next
    raise AttributeError(
        f"Could not find transformer layers for {type(model).__name__}. "
        f"Supported architectures: Llama, Gemma, Qwen, GPT-2/Neo, GPT-NeoX."
    )


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------
#
# This is the central object you create at the start of any run. It holds:
#   - the loaded model (ready for inference)
#   - the tokenizer (turns text into token IDs and back)
#   - the assistant axis (the direction in hidden-state space we cap against)
#   - a reference to the model's layer list (so hooks can be attached)
#
# Everything else in this file operates *on* a SteeringExperiment instance.
# ---------------------------------------------------------------------------

class SteeringExperiment:
    """Loads a transformer model and assistant axis for steering experiments."""

    def __init__(
        self,
        model_name: str,                        # HuggingFace model ID, e.g. "Qwen/Qwen3-32B"
        axis_path: Optional[str] = None,         # path to assistant axis .pt file (None = auto-download)
        device: Optional[str] = None,            # unused for now -- device_map="auto" handles placement
        dtype: torch.dtype = torch.bfloat16,     # half-precision to fit large models in GPU memory
        deterministic: bool = False,             # if True, disable cuDNN non-determinism
    ):
        # Lock down randomness if requested (useful for reproducible experiments)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model_name = model_name
        self._deterministic = deterministic

        # --- Load the model ---
        # device_map="auto" spreads layers across available GPUs automatically.
        # flash_attention_2 is a fast attention implementation that saves memory.
        logger.info("Loading model %s (dtype=%s, device_map=auto)...", model_name, dtype)
        t0 = time.time()
        model_kwargs = dict(dtype=dtype, device_map="auto", attn_implementation="flash_attention_2")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()                        # inference mode -- no gradient tracking
        logger.info("Model loaded in %.1fs", time.time() - t0)

        # --- Load the tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:     # some tokenizers don't define a pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Discover transformer layers ---
        # We need direct access to the layer modules so we can attach hooks later.
        self.layers = _get_layers(self.model)
        self.num_layers = len(self.layers)
        logger.info("Layers: %d, device_map: %s",
                     self.num_layers,
                     dict(self.model.hf_device_map) if hasattr(self.model, "hf_device_map") else "single")

        # --- Load the assistant axis ---
        # If no path given, download the pre-computed axis from HuggingFace.
        if axis_path is None:
            axis_path = download_axis(model_name)
        self.axis = load_axis(axis_path)         # shape: (num_layers, hidden_dim)
        self.hidden_dim = self.axis.shape[-1]
        logger.info("Axis loaded: shape=%s, hidden_dim=%d", self.axis.shape, self.hidden_dim)

    def _model_device(self) -> torch.device:
        """Figure out which device the model's first layer lives on.

        Needed so we can move input tensors to the right GPU.
        """
        if hasattr(self.model, "hf_device_map"):
            first_device = next(iter(self.model.hf_device_map.values()))
            return torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
        return next(self.model.parameters()).device

    def get_baseline_trajectory(
        self, input_ids: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Run a single forward pass (no capping) and record what the hidden state
        looks like at the last token position of every layer.

        This is used during warmup to collect the activations we need for building
        the compliance axis and for computing thresholds.

        Returns:
            activations: dict mapping layer_idx -> hidden state vector (float32, CPU)
            logits:      the model's final logit vector for the last token
        """
        activations = {}
        handles = []

        # Attach a hook to every layer that grabs the last-token hidden state
        for layer_idx in range(self.num_layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    # output is either a tensor or a tuple (hidden_state, ...)
                    act = output[0] if isinstance(output, tuple) else output
                    # [0, -1, :] = first batch item, last token position, all hidden dims
                    activations[idx] = act[0, -1, :].detach().clone().cpu().float()
                return hook_fn
            handles.append(self.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        # Run the forward pass with no gradient computation
        with torch.inference_mode():
            outputs = self.model(input_ids)

        # Clean up hooks so they don't fire during later generation
        for h in handles:
            h.remove()

        logits = outputs.logits[0, -1, :].detach().clone().cpu().float()
        return activations, logits

    def tokenize(self, prompt: str) -> torch.Tensor:
        """Turn a user prompt into token IDs ready for the model.

        Applies the model's chat template (wraps the prompt in the expected
        role markers) and adds the generation prompt so the model knows it
        should start producing an assistant response.

        For Qwen models, we also inject empty <think> tags to skip the
        chain-of-thought reasoning and go straight to the answer.
        """
        conversation = [{"role": "user", "content": prompt}]
        chat_kwargs = {}
        if "qwen" in self.model_name.lower():
            chat_kwargs["enable_thinking"] = False     # tell Qwen not to think out loud
        text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        if "qwen" in self.model_name.lower():
            text += "<think>\n</think>\n\n"            # close the thinking block immediately
        tokens = self.tokenizer(text, return_tensors="pt")
        return tokens["input_ids"].to(self._model_device())


# ---------------------------------------------------------------------------
# Projection tracker  (read-only monitoring hook)
# ---------------------------------------------------------------------------
#
# This hook does NOT change the model's behavior. It just watches.
# At each generation step it records the dot product of the hidden state
# with a given axis, so we can plot how the model's "stance" evolves
# token by token.  Used as a context manager:
#
#   with _AxisProjectionTracker(layer, axis) as tracker:
#       model.generate(...)
#   print(tracker.projections)  # one float per generation step
# ---------------------------------------------------------------------------

class _AxisProjectionTracker:
    """Read-only hook that records the dot product of the last-token hidden state
    with a given axis vector at each forward pass during generation."""

    def __init__(self, layer_module: nn.Module, axis_vector: torch.Tensor):
        self._layer = layer_module           # which transformer layer to watch
        self._axis = axis_vector.float()     # the direction to measure against
        self._axis_device = None             # lazily moved to GPU on first call
        self._projections: list = []         # one entry per generation step
        self._handle = None                  # PyTorch hook handle (for cleanup)

    def __enter__(self):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            h = act[0, -1, :].detach().float()               # last-token hidden state
            if self._axis_device is None:                     # first call: move axis to same GPU
                self._axis_device = self._axis.to(h.device)
            self._projections.append(h @ self._axis_device)   # dot product = projection

        self._handle = self._layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()            # detach hook so it doesn't fire anymore
            self._handle = None

    @property
    def projections(self) -> list[float]:
        """Return all recorded projections as a plain Python list of floats."""
        if not self._projections:
            return []
        return torch.stack(self._projections).cpu().tolist()


# ---------------------------------------------------------------------------
# Capping hooks  (these actually modify the model's hidden states)
# ---------------------------------------------------------------------------
#
# Both hooks work the same way at a high level:
#   1. At each generation step, read the last-token hidden state h.
#   2. Check if h's projection onto the detection axis is below a threshold.
#   3. If yes, nudge h along the correction axis to bring it back up.
#
# The difference between the two classes:
#   _CappingHook          -- detection axis = correction axis (single axis)
#   _CrossAxisCappingHook -- detection axis != correction axis (two axes)
# ---------------------------------------------------------------------------

class _CappingHook:
    """Single-axis capping: detect AND correct on the same axis.

    At every generation step:
      1. Compute proj = dot(h, v)           -- how far along the axis is h?
      2. If proj < tau (threshold):
           nudge = (tau - proj) * v          -- just enough to reach the threshold
           h = h + nudge                     -- modify the hidden state in place
      3. Otherwise do nothing.

    This is the standard intervention from prior work.

    Formula:  h <- h - v * min(<h, v> - tau, 0)
    """

    def __init__(
        self,
        layer_module: nn.Module,       # the transformer layer to hook into
        axis_unit: torch.Tensor,       # unit vector defining the capping direction
        threshold: float,              # projection values below this trigger correction
    ):
        self._layer = layer_module
        self._axis = axis_unit.float()
        self._tau = threshold
        self._axis_device: Optional[torch.Tensor] = None   # lazily moved to GPU
        self.n_interventions = 0       # counts how many times capping actually fired
        self._handle = None

    def __enter__(self):
        def hook_fn(module, input, output):
            # Unpack -- some models return a tuple, others a raw tensor
            if torch.is_tensor(output):
                h = output
            else:
                h = output[0]

            # Move axis to the same device as h on first call
            if self._axis_device is None:
                self._axis_device = self._axis.to(h.device)

            # Cap at every token position (matching the original paper).
            # During generation with KV-cache, seq_len is 1 (just the new token).
            # During the prefill pass, seq_len is the full prompt length.
            axis = self._axis_device.float()
            h_float = h[0].float()                          # (seq_len, hidden_dim)
            projs = h_float @ axis                           # (seq_len,) -- projection per position
            mask = projs < self._tau                          # which positions need capping?
            if mask.any():
                deltas = (self._tau - projs[mask]).unsqueeze(-1) * axis  # (n_below, hidden_dim)
                h[0, mask] += deltas.to(h.dtype)
                self.n_interventions += int(mask.sum().item())

            # Re-pack into the original output format
            if torch.is_tensor(output):
                return h
            return (h, *output[1:])

        self._handle = self._layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class _CrossAxisCappingHook:
    """Cross-axis capping: detect on one axis, correct along a DIFFERENT axis.

    This is the novel part of the experiment. The idea is that the axis
    which best *detects* jailbreaks (the assistant axis) may not be the
    best axis to *correct* along. So we split the roles:

      Detection axis  (e.g. assistant axis):
        "Is the model about to comply with a harmful request?"
        Fires when dot(h, v_detect) < tau_detect.

      Correction axis (e.g. PCA compliance axis):
        "Push the model toward refusal."
        Applies h <- h + (tau_correct - proj_correct) * v_correct,
        but ONLY when the detection axis triggered first.

    This two-gate design means:
      - Detection is still handled by the well-studied assistant axis
      - Correction can use a purpose-built direction that may be more
        effective at inducing refusal without degrading benign outputs
    """

    def __init__(
        self,
        layer_module: nn.Module,
        detect_axis: torch.Tensor,       # axis used to decide IF we should intervene
        detect_threshold: float,         # fire when projection falls below this
        correct_axis: torch.Tensor,      # axis used to perform the actual correction
        correct_threshold: float,        # target projection value on the correction axis
    ):
        self._layer = layer_module
        self._detect_axis = detect_axis.float()
        self._tau_detect = detect_threshold
        self._correct_axis = correct_axis.float()
        self._tau_correct = correct_threshold
        self._detect_device: Optional[torch.Tensor] = None    # lazily moved to GPU
        self._correct_device: Optional[torch.Tensor] = None
        self.n_triggered = 0             # detection axis fired (gate opened)
        self.n_corrected = 0             # correction was also needed and applied
        self._handle = None

    def __enter__(self):
        def hook_fn(module, input, output):
            # Unpack output
            if torch.is_tensor(output):
                h = output
            else:
                h = output[0]

            h_last = h[0, -1, :].float()     # last-token hidden state

            # Move both axes to the right GPU on first call
            if self._detect_device is None:
                self._detect_device = self._detect_axis.to(h.device)
                self._correct_device = self._correct_axis.to(h.device)

            # --- GATE 1: Detection ---
            # Check the assistant axis: is the model heading toward compliance?
            detect_proj = (h_last @ self._detect_device).item()
            if detect_proj < self._tau_detect:
                self.n_triggered += 1        # yes, detection fired

                # --- GATE 2: Correction ---
                # Now check the *correction* axis: does h also need nudging there?
                correct_proj = (h_last @ self._correct_device).item()
                if correct_proj < self._tau_correct:
                    # Push h along the correction axis to reach the threshold
                    delta = (self._tau_correct - correct_proj) * self._correct_device.to(h.dtype)
                    h[0, -1, :].add_(delta)  # in-place modification
                    self.n_corrected += 1

            # Re-pack into the original output format
            if torch.is_tensor(output):
                return h
            return (h, *output[1:])

        self._handle = self._layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------
#
# Before we can cap, we need to know *when* to fire. The idea:
#   1. Run a batch of benign prompts and record their projection values.
#   2. Run a batch of jailbreak prompts and record theirs.
#   3. Set the threshold at the midpoint between the two distributions.
#
# This way, benign prompts (which project high on the assistant axis)
# mostly stay above the threshold (no capping), while jailbreak prompts
# (which project low) fall below it and get corrected.
# ---------------------------------------------------------------------------

def _collect_projections(
    exp: SteeringExperiment,
    prompts: list[str],
    axis_directions: dict[str, dict[int, torch.Tensor]],
    cap_layers: list[int],
    max_new_tokens: int = 64,
    label: str = "",
) -> dict[str, dict[int, list[float]]]:
    """Run prompts through the model and record projection values at each
    cap layer for each axis.

    This is a helper used by compute_discriminative_thresholds -- it
    generates text for each prompt while tracking projections, then
    returns all the recorded values so we can compute statistics.

    Args:
        axis_directions: maps axis_name -> layer_idx -> unit vector for that layer.

    Returns:
        projs[axis_name][layer_idx] = list of floats (one per generation step
        across all prompts).
    """
    # Pre-allocate empty lists for every (axis, layer) combination
    projs: dict[str, dict[int, list[float]]] = {
        name: {layer_idx: [] for layer_idx in cap_layers}
        for name in axis_directions
    }
    desc = f"  {label}" if label else "Collecting"
    for prompt in tqdm(prompts, desc=desc, leave=False):
        input_ids = exp.tokenize(prompt)

        # Attach a tracker to every (axis, layer) pair, using the
        # layer-specific axis vector for each
        with ExitStack() as stack:
            trackers: dict[tuple, _AxisProjectionTracker] = {}
            for axis_name, per_layer in axis_directions.items():
                for layer_idx in cap_layers:
                    t = _AxisProjectionTracker(exp.layers[layer_idx], per_layer[layer_idx])
                    stack.enter_context(t)
                    trackers[(axis_name, layer_idx)] = t

            # Generate (we don't need the output text, just the projections)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                exp.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        # Collect all recorded projections from this prompt
        for (axis_name, layer_idx), tracker in trackers.items():
            projs[axis_name][layer_idx].extend(tracker.projections)
    return projs


def compute_discriminative_thresholds(
    exp: SteeringExperiment,
    benign_prompts: list[str],
    jailbreak_prompts: list[str],
    axis_directions: dict[str, dict[int, torch.Tensor]],
    cap_layers: list[int],
    max_new_tokens: int = 64,
) -> dict[str, dict[int, dict[str, float]]]:




    """Find the per-layer threshold that best separates benign from jailbreak
    prompts on each axis.

    Strategy: set tau = midpoint of (mean_benign, mean_jailbreak). This is
    a simple linear discriminant -- prompts whose projection falls below tau
    will be treated as potential jailbreaks and get capped.

    Also records stats (means, stds, separation) so the caller can log them
    and save them to metadata.

    Returns:
        thresholds[axis_name][layer_idx] = {
            "optimal":        the midpoint threshold (tau),
            "mean_benign":    average projection for benign prompts,
            "mean_jailbreak": average projection for jailbreak prompts,
            "std_benign":     standard deviation for benign,
            "std_jailbreak":  standard deviation for jailbreak,
            "separation":     gap between the two means (bigger = easier to tell apart),
        }
    """




    logger.info(
        "Computing discriminative thresholds at L%d-L%d "
        "(%d benign, %d jailbreak prompts)...",
        cap_layers[0], cap_layers[-1],
        len(benign_prompts), len(jailbreak_prompts),
    )

    # Step 1: collect projection values from both prompt sets
    benign_projs    = _collect_projections(exp, benign_prompts,    axis_directions, cap_layers, max_new_tokens, label="Benign")
    jailbreak_projs = _collect_projections(exp, jailbreak_prompts, axis_directions, cap_layers, max_new_tokens, label="Jailbreak")

    # Step 2: for each (axis, layer), compute the midpoint threshold
    thresholds: dict[str, dict[int, dict[str, float]]] = {}
    for axis_name in axis_directions:
        thresholds[axis_name] = {}
        for layer_idx in cap_layers:
            b_arr = np.array(benign_projs[axis_name][layer_idx],   dtype=np.float32)
            j_arr = np.array(jailbreak_projs[axis_name][layer_idx], dtype=np.float32)
            mean_b, mean_j = float(b_arr.mean()), float(j_arr.mean())
            tau = (mean_b + mean_j) / 2.0     # midpoint = decision boundary
            separation = mean_b - mean_j       # how well-separated the two classes are
            combined = np.concatenate([b_arr, j_arr])
            thresholds[axis_name][layer_idx] = {
                "optimal":        tau,
                "p25":            float(np.percentile(combined, 25)),
                "mean_benign":    mean_b,
                "mean_jailbreak": mean_j,
                "std_benign":     float(b_arr.std()),
                "std_jailbreak":  float(j_arr.std()),
                "separation":     separation,
            }

        # Log the first and last cap layers for a quick sanity check
        for layer_idx in [cap_layers[0], cap_layers[-1]]:
            d = thresholds[axis_name][layer_idx]
            logger.info(
                "  %s L%d: benign=%.1f+/-%.1f  jailbreak=%.1f+/-%.1f  "
                "sep=%.1f  tau=%.1f",
                axis_name, layer_idx,
                d["mean_benign"], d["std_benign"],
                d["mean_jailbreak"], d["std_jailbreak"],
                d["separation"], d["optimal"],
            )

    return thresholds


def compute_cross_detect_thresholds(
    exp: "SteeringExperiment",
    benign_prompts: list[str],
    jailbreak_prompts: list[str],
    assistant_axes: dict[int, torch.Tensor],
    cap_layers: list[int],
    method: str = "benign-p5",
) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    """Per-layer detection thresholds for the cross-cap gate, computed from
    your own benign + jailbreak calibration prompts projected onto the
    (paper's) assistant axis.

    The assistant axis vector is unchanged -- only the scalar cutoff used by
    the cross-cap detection gate is recomputed, so the gate fires on prompts
    that look jailbreak-like on YOUR eval distribution rather than on the
    paper's calibration distribution. The paper's assistant_taus are kept
    untouched for Mode 2 (single-axis assistant-cap), so the baseline
    comparison stays fair.

    Detection fires when (h_last @ v_assist) < tau. Lower tau = fires less
    often (less aggressive), which is the point of tuning this cutoff.

    Methods (all data-driven on YOUR prompts):
      - "benign-p5":  tau = 5th percentile of benign projections
                      (by construction <= 5% of benign prompts trip
                      detection, regardless of where jailbreaks land)
      - "benign-p10": 10th percentile of benign projections (looser)
      - "midpoint":   tau = (mean_benign + mean_jailbreak) / 2
                      (symmetric discriminative boundary)

    Returns:
        taus:  {layer_idx -> tau (float)}
        stats: {layer_idx -> {
            "mean_benign", "std_benign",
            "mean_jailbreak", "std_jailbreak",
            "p5_benign", "p10_benign", "midpoint", "separation",
        }}
    """
    logger.info(
        "Computing cross-cap detection thresholds on assistant axis at "
        "L%d-L%d (%d benign, %d jailbreak, method=%s)...",
        cap_layers[0], cap_layers[-1],
        len(benign_prompts), len(jailbreak_prompts), method,
    )

    def last_token_projs(prompts: list[str], label: str) -> dict[int, list[float]]:
        projs = {li: [] for li in cap_layers}
        for prompt in tqdm(prompts, desc=f"  detect-cal {label}", leave=False):
            ids = exp.tokenize(prompt)
            hs, _ = exp.get_baseline_trajectory(ids)
            for li in cap_layers:
                h_last = hs[li].float()
                v = assistant_axes[li].float().to(h_last.device)
                projs[li].append((h_last @ v).item())
        return projs

    benign_projs = last_token_projs(benign_prompts, "benign")
    jb_projs     = last_token_projs(jailbreak_prompts, "jailbreak")

    taus: dict[int, float] = {}
    stats: dict[int, dict[str, float]] = {}
    for li in cap_layers:
        b = np.asarray(benign_projs[li], dtype=np.float32)
        j = np.asarray(jb_projs[li],     dtype=np.float32)
        mean_b, mean_j = float(b.mean()), float(j.mean())
        p5_b  = float(np.percentile(b,  5))
        p10_b = float(np.percentile(b, 10))
        midpoint = (mean_b + mean_j) / 2.0
        stats[li] = {
            "mean_benign":    mean_b,
            "std_benign":     float(b.std()),
            "mean_jailbreak": mean_j,
            "std_jailbreak":  float(j.std()),
            "p5_benign":      p5_b,
            "p10_benign":     p10_b,
            "midpoint":       midpoint,
            "separation":     mean_b - mean_j,
        }
        if method == "benign-p5":
            taus[li] = p5_b
        elif method == "benign-p10":
            taus[li] = p10_b
        elif method == "midpoint":
            taus[li] = midpoint
        else:
            raise ValueError(
                f"Unknown cross-detect method: {method!r} "
                "(expected benign-p5, benign-p10, or midpoint)"
            )

    for li in [cap_layers[0], cap_layers[-1]]:
        s = stats[li]
        logger.info(
            "  detect-tau L%d: tau=%.2f  benign=%.2f+/-%.2f  jailbreak=%.2f+/-%.2f  sep=%.2f",
            li, taus[li],
            s["mean_benign"], s["std_benign"],
            s["mean_jailbreak"], s["std_jailbreak"],
            s["separation"],
        )

    return taus, stats


# ---------------------------------------------------------------------------
# Compliance axis construction
# ---------------------------------------------------------------------------
#
# The assistant axis (loaded from HuggingFace) is good at *detecting*
# jailbreaks, but it wasn't designed to be the best direction to *correct*
# along. These functions build a second axis -- the "compliance axis" --
# that captures the direction in hidden-state space separating the model's
# "I will refuse" activations from its "I will comply" activations.
#
# Two construction methods are provided:
#   - PCA:       pool refusing + compliant activations, take PC1
#   - Mean-diff: simply (mean_refusing - mean_compliant), normalised
#
# PCA is the default because it captures the dominant direction of variation
# without assuming the two clusters have equal spread.
# ---------------------------------------------------------------------------

def compute_pca_compliance_axis(
    exp: SteeringExperiment,
    refusing_prompts: list[str],       # prompts the model refuses (e.g. bare harmful goals)
    compliant_prompts: list[str],      # prompts the model complies with (e.g. jailbreaks that work)
    cap_layers: list[int],
    axis_name: str = "pca_compliance",
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    """Build per-layer compliance axes via PCA and compute thresholds.

    Runs PCA separately at each cap layer, producing a different compliance
    direction for each layer. Also computes threshold stats from the same
    refusing/compliant activations used to build the axis — no separate
    calibration data needed.

    Returns:
        per_layer_axes:  dict mapping layer_idx -> unit vector on CPU
        per_layer_stats: dict mapping layer_idx -> {
            "mean_refusing", "mean_compliant", "std_refusing", "std_compliant",
            "optimal" (midpoint), "mean+std" (mean_compliant + std_compliant),
            "separation", "var_explained"
        }
    """
    logger.info(
        "Computing %s PCA compliance axes at L%d-L%d (%d refusing, %d compliant)...",
        axis_name, cap_layers[0], cap_layers[-1],
        len(refusing_prompts), len(compliant_prompts),
    )

    # --- Collect activations at ALL cap layers in one pass ---
    refusing_acts = {li: [] for li in cap_layers}
    compliant_acts = {li: [] for li in cap_layers}

    for prompt in tqdm(refusing_prompts, desc=f"  {axis_name} refusing", leave=False):
        ids = exp.tokenize(prompt)
        acts, _ = exp.get_baseline_trajectory(ids)
        for li in cap_layers:
            refusing_acts[li].append(acts[li].float())

    for prompt in tqdm(compliant_prompts, desc=f"  {axis_name} compliant", leave=False):
        ids = exp.tokenize(prompt)
        acts, _ = exp.get_baseline_trajectory(ids)
        for li in cap_layers:
            compliant_acts[li].append(acts[li].float())

    # --- Run PCA at each cap layer separately ---
    per_layer_axes = {}
    per_layer_stats = {}
    for li in cap_layers:
        refusing_stack  = torch.stack(refusing_acts[li])
        compliant_stack = torch.stack(compliant_acts[li])
        pooled = torch.cat([refusing_stack, compliant_stack], dim=0)

        pooled_centered = pooled - pooled.mean(dim=0)
        _, S, Vt = torch.linalg.svd(pooled_centered, full_matrices=False)
        pc1 = Vt[0].float()

        # Fix sign so PC1 points from compliant -> refusing
        mean_diff = refusing_stack.mean(0) - compliant_stack.mean(0)
        if (pc1 @ mean_diff).item() < 0:
            pc1 = -pc1

        pc1 = pc1 / pc1.norm()
        var_explained = (S[0] ** 2 / (S ** 2).sum()).item() * 100
        per_layer_axes[li] = pc1.cpu()

        # --- Compute threshold stats from refusing/compliant projections ---
        # On this axis: high = refusing (safe), low = compliant (unsafe)
        refusing_projs = (refusing_stack @ pc1).numpy()
        compliant_projs = (compliant_stack @ pc1).numpy()
        mean_r = float(refusing_projs.mean())
        mean_c = float(compliant_projs.mean())
        std_r = float(refusing_projs.std())
        std_c = float(compliant_projs.std())
        per_layer_stats[li] = {
            "mean_refusing":  mean_r,
            "mean_compliant": mean_c,
            "std_refusing":   std_r,
            "std_compliant":  std_c,
            "optimal":        (mean_r + mean_c) / 2.0,
            "mean+std":       mean_c + std_c,
            "separation":     mean_r - mean_c,
            "var_explained":  var_explained,
        }
        logger.info(
            "  %s L%d: var=%.1f%%  refusing=%.1f+/-%.1f  compliant=%.1f+/-%.1f  sep=%.1f",
            axis_name, li, var_explained, mean_r, std_r, mean_c, std_c,
            mean_r - mean_c,
        )

    return per_layer_axes, per_layer_stats, refusing_acts, compliant_acts


def compute_mean_diff_compliance_axis(
    exp: SteeringExperiment,
    refusing_prompts: list[str],
    compliant_prompts: list[str],
    cap_layers: list[int],
    axis_name: str = "mean_diff_compliance",
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    """Simpler alternative: compliance axis = (mean_refusing - mean_compliant).

    Computed per-layer, with threshold stats derived from the same data.
    Less principled than PCA but faster and sometimes works just as well.
    Each result is L2-normalised to unit length.

    Returns:
        per_layer_axes:  dict mapping layer_idx -> unit vector on CPU
        per_layer_stats: dict mapping layer_idx -> threshold stats (same format as PCA)
    """
    logger.info(
        "Computing %s mean-diff compliance axes at L%d-L%d (%d refusing, %d compliant)...",
        axis_name, cap_layers[0], cap_layers[-1],
        len(refusing_prompts), len(compliant_prompts),
    )

    refusing_acts = {li: [] for li in cap_layers}
    compliant_acts = {li: [] for li in cap_layers}

    for prompt in tqdm(refusing_prompts, desc=f"  {axis_name} refusing", leave=False):
        ids = exp.tokenize(prompt)
        acts, _ = exp.get_baseline_trajectory(ids)
        for li in cap_layers:
            refusing_acts[li].append(acts[li].float())

    for prompt in tqdm(compliant_prompts, desc=f"  {axis_name} compliant", leave=False):
        ids = exp.tokenize(prompt)
        acts, _ = exp.get_baseline_trajectory(ids)
        for li in cap_layers:
            compliant_acts[li].append(acts[li].float())

    per_layer_axes = {}
    per_layer_stats = {}
    for li in cap_layers:
        refusing_stack = torch.stack(refusing_acts[li])
        compliant_stack = torch.stack(compliant_acts[li])
        md = refusing_stack.mean(0) - compliant_stack.mean(0)
        md = md / md.norm()
        per_layer_axes[li] = md.cpu()

        # Threshold stats from projections onto this axis
        refusing_projs = (refusing_stack @ md).numpy()
        compliant_projs = (compliant_stack @ md).numpy()
        mean_r = float(refusing_projs.mean())
        mean_c = float(compliant_projs.mean())
        std_r = float(refusing_projs.std())
        std_c = float(compliant_projs.std())
        per_layer_stats[li] = {
            "mean_refusing":  mean_r,
            "mean_compliant": mean_c,
            "std_refusing":   std_r,
            "std_compliant":  std_c,
            "optimal":        (mean_r + mean_c) / 2.0,
            "mean+std":       mean_c + std_c,
            "separation":     mean_r - mean_c,
        }
        logger.info(
            "  %s L%d: refusing=%.1f+/-%.1f  compliant=%.1f+/-%.1f  sep=%.1f",
            axis_name, li, mean_r, std_r, mean_c, std_c, mean_r - mean_c,
        )

    return per_layer_axes, per_layer_stats, refusing_acts, compliant_acts


def orthogonalize_compliance_axes(
    exp: SteeringExperiment,
    compliance_axes: dict[int, torch.Tensor],
    benign_prompts: list[str],
    refusing_acts: dict[int, list[torch.Tensor]],
    compliant_acts: dict[int, list[torch.Tensor]],
    cap_layers: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    """Remove the benign component from compliance axes.

    Projects out the mean benign activation direction from each compliance
    axis. The resulting vector can still push toward refusal but has zero
    projection onto the direction that benign prompts naturally occupy.

    Reuses the refusing/compliant activations already collected during
    axis construction to recompute threshold stats on the new axes.

    Returns:
        orth_axes:       dict mapping layer_idx -> orthogonalized unit vector
        orth_stats:      dict mapping layer_idx -> threshold stats
    """
    logger.info(
        "Orthogonalizing compliance axes against benign direction "
        "(%d benign prompts, L%d-L%d)...",
        len(benign_prompts), cap_layers[0], cap_layers[-1],
    )

    # Only need to collect benign activations — refusing/compliant are reused
    benign_acts = {li: [] for li in cap_layers}
    for prompt in tqdm(benign_prompts, desc="  Benign activations", leave=False):
        ids = exp.tokenize(prompt)
        acts, _ = exp.get_baseline_trajectory(ids)
        for li in cap_layers:
            benign_acts[li].append(acts[li].float())

    orth_axes = {}
    orth_stats = {}
    for li in cap_layers:
        benign_stack = torch.stack(benign_acts[li])

        # Benign direction = mean benign activation (normalized).
        # This is the direction we want to protect from capping.
        benign_dir = benign_stack.mean(dim=0).float()
        benign_dir = benign_dir / benign_dir.norm()

        # Orthogonalize: remove benign component from compliance axis
        comp = compliance_axes[li].float()
        overlap = (comp @ benign_dir).item()
        orth = comp - overlap * benign_dir
        orth = orth / orth.norm()
        orth_axes[li] = orth.cpu()

        # Recompute threshold stats on the orthogonalized axis
        refusing_stack = torch.stack(refusing_acts[li])
        compliant_stack = torch.stack(compliant_acts[li])
        refusing_projs = (refusing_stack @ orth).numpy()
        compliant_projs = (compliant_stack @ orth).numpy()
        mean_r = float(refusing_projs.mean())
        mean_c = float(compliant_projs.mean())
        std_r = float(refusing_projs.std())
        std_c = float(compliant_projs.std())
        orth_stats[li] = {
            "mean_refusing":  mean_r,
            "mean_compliant": mean_c,
            "std_refusing":   std_r,
            "std_compliant":  std_c,
            "optimal":        (mean_r + mean_c) / 2.0,
            "mean+std":       mean_c + std_c,
            "separation":     mean_r - mean_c,
        }

        logger.info(
            "  L%d: cos(comp,benign)=%.3f -> %.6f  sep=%.1f",
            li, overlap, (orth @ benign_dir).item(), mean_r - mean_c,
        )

    return orth_axes, orth_stats


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------
#
# These are the three functions that run_crosscap.py calls for each prompt.
# They all follow the same pattern:
#   1. Set up hooks (trackers for monitoring, capping hooks for intervention)
#   2. Call model.generate() to produce text token by token
#   3. Collect diagnostics (projections, intervention counts, which layers fired)
#   4. Return everything so the orchestrator can write it to CSV
#
# The three modes:
#   generate_baseline      -- no capping, just observe (the control condition)
#   generate_capped        -- single-axis capping (the existing method)
#   generate_cross_capped  -- cross-axis capping (the new method we're testing)
# ---------------------------------------------------------------------------

def generate_baseline(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    axis_directions: dict[str, dict[int, torch.Tensor]],   # axes to monitor (not used for capping)
    track_layers: list[int],                     # which layers to record projections at
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = False,                     # False = greedy decoding (deterministic)
) -> tuple:


    """Generate text with NO capping -- this is the control condition.

    We still attach projection trackers so we can compare how the model's
    internal state differs between capped and uncapped runs.

    Args:
        axis_directions: maps axis_name -> layer_idx -> unit vector for that layer.

    Returns:
        sequences:  the generated token IDs
        scores:     per-step logit distributions
        projs:      projs[axis_name][layer_idx] = list of projection values,
                    one per generation step
    """


    with ExitStack() as stack:
        # Attach read-only trackers to monitor projections (no intervention)
        trackers: dict[tuple, _AxisProjectionTracker] = {}
        for axis_name, per_layer in axis_directions.items():
            for layer_idx in track_layers:
                t = _AxisProjectionTracker(exp.layers[layer_idx], per_layer[layer_idx])
                stack.enter_context(t)
                trackers[(axis_name, layer_idx)] = t

        # Standard generation setup
        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,              # we want the raw logits at each step
            return_dict_in_generate=True,    # return a structured object, not just token IDs
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature)

        # Actually generate -- hooks fire automatically at each step
        with torch.inference_mode():
            output = exp.model.generate(input_ids, **gen_kwargs)

        # Collect all recorded projections into a nested dict
        projs = {
            axis_name: {
                layer_idx: trackers[(axis_name, layer_idx)].projections
                for layer_idx in track_layers
            }
            for axis_name in axis_directions
        }

    return output.sequences, output.scores, projs


def generate_capped(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    cap_layers: list[int],                       # which layers to apply capping at
    per_layer_axes: dict[int, torch.Tensor],     # per-layer assistant axis vectors
    per_layer_thresholds: dict[int, float],       # threshold per layer
    track_layers: list[int],                     # which layers to track projections at
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> tuple:


    """Generate text with SINGLE-AXIS capping active.

    This is the baseline capping method: at each generation step, every
    cap layer checks if the hidden state's projection onto the assistant axis
    is below threshold, and nudges it back up if so. Each layer uses its own
    axis vector (matching the original paper's per-layer extraction).

    Returns:
        sequences:        generated token IDs
        scores:           per-step logit distributions
        projs:            projection traces per layer
        n_interventions:  total number of times any layer fired
        active_layers:    list of layer indices that fired at least once
    """



    # Create one capping hook per cap layer, each with its own axis and threshold
    cap_hooks = [
        _CappingHook(exp.layers[layer_idx], per_layer_axes[layer_idx], per_layer_thresholds[layer_idx])
        for layer_idx in cap_layers
    ]

    with ExitStack() as stack:
        # Activate all capping hooks
        for hook in cap_hooks:
            stack.enter_context(hook)

        # Also attach read-only trackers so we can see the post-cap projections
        trackers: dict[int, _AxisProjectionTracker] = {}
        for layer_idx in track_layers:
            t = _AxisProjectionTracker(exp.layers[layer_idx], per_layer_axes[layer_idx])
            stack.enter_context(t)
            trackers[layer_idx] = t

        # Generate
        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature)
        with torch.inference_mode():
            output = exp.model.generate(input_ids, **gen_kwargs)

        projs = {layer_idx: trackers[layer_idx].projections for layer_idx in track_layers}

    # Summarise what happened: how many interventions, and at which layers?
    n_interventions = sum(h.n_interventions for h in cap_hooks)
    active_layers = [li for li, h in zip(cap_layers, cap_hooks) if h.n_interventions > 0]
    return output.sequences, output.scores, projs, n_interventions, active_layers


def generate_cross_capped(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    cap_layers: list[int],
    per_layer_detect_axes: dict[int, torch.Tensor],  # per-layer assistant axes for detection
    correct_axes: dict[int, torch.Tensor],       # per-layer compliance axes for correction
    detect_thresholds: dict[int, float],          # per-layer thresholds for detection
    correct_thresholds: dict[int, float],         # per-layer thresholds for correction
    track_layers: list[int],
    per_layer_track_axes: dict[int, torch.Tensor],   # per-layer axes for monitoring
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> tuple:


    """Generate text with CROSS-AXIS capping: detect on one axis, correct on another.

    This is the novel method. At each generation step and each cap layer:
      1. Check the assistant axis -- is the projection below threshold? (detection)
      2. If yes, check the compliance axis -- is correction needed? (gating)
      3. If yes, nudge the hidden state along the compliance axis. (correction)

    Each layer uses its own assistant axis vector for detection (matching the
    original paper's per-layer extraction).

    Returns:
        sequences:        generated token IDs
        scores:           per-step logit distributions
        projs:            projection traces per layer
        n_triggered:      how many times the detection gate opened
        n_corrected:      how many times correction was actually applied
        corrected_layers: which layers applied at least one correction
    """



    # Create one cross-axis hook per cap layer
    hooks = [
        _CrossAxisCappingHook(
            exp.layers[layer_idx],
            per_layer_detect_axes[layer_idx], detect_thresholds[layer_idx],  # gate: assistant axis
            correct_axes[layer_idx], correct_thresholds[layer_idx],          # correction: compliance axis
        )
        for layer_idx in cap_layers
    ]

    with ExitStack() as stack:
        # Activate all cross-axis hooks
        for hook in hooks:
            stack.enter_context(hook)

        # Attach trackers for monitoring
        trackers: dict[int, _AxisProjectionTracker] = {}
        for layer_idx in track_layers:
            t = _AxisProjectionTracker(exp.layers[layer_idx], per_layer_track_axes[layer_idx])
            stack.enter_context(t)
            trackers[layer_idx] = t

        # Generate
        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature)
        with torch.inference_mode():
            output = exp.model.generate(input_ids, **gen_kwargs)

        projs = {layer_idx: trackers[layer_idx].projections for layer_idx in track_layers}

    # Summarise: triggered = detection fired, corrected = correction actually applied
    # (triggered >= corrected, because detection can fire without correction being needed)
    n_triggered = sum(h.n_triggered for h in hooks)
    n_corrected = sum(h.n_corrected for h in hooks)
    corrected_layers = [li for li, h in zip(cap_layers, hooks) if h.n_corrected > 0]
    return output.sequences, output.scores, projs, n_triggered, n_corrected, corrected_layers
