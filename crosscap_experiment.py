"""
Cross-axis capping experiment library.

Two capping modes:
  - Assistant-axis capping: detect and correct on the same (assistant) axis.
  - Cross-axis capping: detect on assistant axis, correct on PCA compliance axis.

Formula:  h <- h - v * min(<h, v> - tau, 0)

Pure library module -- all orchestration lives in run_crosscap.py.
"""

import logging
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from contextlib import ExitStack
from typing import Optional
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("crosscap")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Model & axis configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "google/gemma-2-27b-it": {
        "hf_axis_file": "gemma-2-27b/assistant_axis.pt",
        "target_layer": 22,
        "num_layers": 46,
    },
    "Qwen/Qwen3-32B": {
        "hf_axis_file": "qwen-3-32b/assistant_axis.pt",
        "target_layer": 32,
        "num_layers": 64,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "hf_axis_file": "llama-3.3-70b/assistant_axis.pt",
        "target_layer": 40,
        "num_layers": 80,
    },
}

HF_AXIS_REPO = "lu-christina/assistant-axis-vectors"


def load_axis(path: str) -> torch.Tensor:
    """Load a pre-computed assistant axis from a .pt file.

    Handles both formats: dict with 'axis' key, or raw tensor.
    Returns tensor of shape (num_layers, hidden_dim).
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "axis" in data:
            return data["axis"]
        raise ValueError("Expected 'axis' key in saved dict")
    return data


def download_axis(model_name: str, cache_dir: str = "results") -> str:
    """Download pre-computed assistant axis from HuggingFace."""
    cfg = MODEL_CONFIGS[model_name]
    return hf_hub_download(
        repo_id=HF_AXIS_REPO,
        filename=cfg["hf_axis_file"],
        repo_type="dataset",
        local_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------

_LAYER_PATHS = [
    lambda m: m.model.layers,            # Llama, Gemma 2, Qwen, Mistral
    lambda m: m.language_model.layers,    # Gemma 3, LLaVA (vision-language)
    lambda m: m.transformer.h,            # GPT-2/Neo, Bloom
    lambda m: m.transformer.layers,       # Some transformer variants
    lambda m: m.gpt_neox.layers,          # GPT-NeoX
]


def _get_layers(model: nn.Module) -> nn.ModuleList:
    """Find the transformer layer list for any supported architecture."""
    for path_fn in _LAYER_PATHS:
        try:
            layers = path_fn(model)
            if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise AttributeError(
        f"Could not find transformer layers for {type(model).__name__}. "
        f"Supported architectures: Llama, Gemma, Qwen, GPT-2/Neo, GPT-NeoX."
    )


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class SteeringExperiment:
    """Loads a transformer model and assistant axis for steering experiments."""

    def __init__(
        self,
        model_name: str,
        axis_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        deterministic: bool = False,
    ):
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model_name = model_name
        self._deterministic = deterministic

        logger.info("Loading model %s (dtype=%s, device_map=auto)...", model_name, dtype)
        t0 = time.time()
        model_kwargs = dict(dtype=dtype, device_map="auto", attn_implementation="flash_attention_2")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        logger.info("Model loaded in %.1fs", time.time() - t0)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.layers = _get_layers(self.model)
        self.num_layers = len(self.layers)
        logger.info("Layers: %d, device_map: %s",
                     self.num_layers,
                     dict(self.model.hf_device_map) if hasattr(self.model, "hf_device_map") else "single")

        if axis_path is None:
            axis_path = download_axis(model_name)
        self.axis = load_axis(axis_path)  # (num_layers, hidden_dim)
        self.hidden_dim = self.axis.shape[-1]
        logger.info("Axis loaded: shape=%s, hidden_dim=%d", self.axis.shape, self.hidden_dim)

    def _model_device(self) -> torch.device:
        if hasattr(self.model, "hf_device_map"):
            first_device = next(iter(self.model.hf_device_map.values()))
            return torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
        return next(self.model.parameters()).device

    def get_baseline_trajectory(
        self, input_ids: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Run unperturbed forward pass and cache residual stream at all layers."""
        activations = {}
        handles = []
        for layer_idx in range(self.num_layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    act = output[0] if isinstance(output, tuple) else output
                    activations[idx] = act[0, -1, :].detach().clone().cpu().float()
                return hook_fn
            handles.append(self.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
        with torch.inference_mode():
            outputs = self.model(input_ids)
        for h in handles:
            h.remove()
        logits = outputs.logits[0, -1, :].detach().clone().cpu().float()
        return activations, logits

    def tokenize(self, prompt: str) -> torch.Tensor:
        """Tokenize a prompt with chat template applied, return input_ids on device."""
        conversation = [{"role": "user", "content": prompt}]
        chat_kwargs = {}
        if "qwen" in self.model_name.lower():
            chat_kwargs["enable_thinking"] = False
        text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        if "qwen" in self.model_name.lower():
            text += "<think>\n</think>\n\n"
        tokens = self.tokenizer(text, return_tensors="pt")
        return tokens["input_ids"].to(self._model_device())


# ---------------------------------------------------------------------------
# Projection tracker
# ---------------------------------------------------------------------------

class _AxisProjectionTracker:
    """Read-only hook that records the dot product of the last-token hidden state
    with a given axis vector at each forward pass during generation."""

    def __init__(self, layer_module: nn.Module, axis_vector: torch.Tensor):
        self._layer = layer_module
        self._axis = axis_vector.float()
        self._axis_device = None
        self._projections: list = []
        self._handle = None

    def __enter__(self):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            h = act[0, -1, :].detach().float()
            if self._axis_device is None:
                self._axis_device = self._axis.to(h.device)
            self._projections.append(h @ self._axis_device)

        self._handle = self._layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def projections(self) -> list[float]:
        if not self._projections:
            return []
        return torch.stack(self._projections).cpu().tolist()


# ---------------------------------------------------------------------------
# Capping hooks
# ---------------------------------------------------------------------------

class _CappingHook:
    """Context manager enforcing a minimum projection threshold at one layer.

    Fires on every forward pass. If the last-token hidden state's projection
    onto axis_unit is below threshold tau, adds exactly enough to bring it to tau.

    Formula:  h <- h - v * min(<h, v> - tau, 0)
    """

    def __init__(
        self,
        layer_module: nn.Module,
        axis_unit: torch.Tensor,
        threshold: float,
    ):
        self._layer = layer_module
        self._axis = axis_unit.float()
        self._tau = threshold
        self._axis_device: Optional[torch.Tensor] = None
        self.n_interventions = 0
        self._handle = None

    def __enter__(self):
        def hook_fn(module, input, output):
            if torch.is_tensor(output):
                h = output
            else:
                h = output[0]

            if self._axis_device is None:
                self._axis_device = self._axis.to(h.device)

            proj = (h[0, -1, :].float() @ self._axis_device.float()).item()

            if proj < self._tau:
                delta = (self._tau - proj) * self._axis_device.to(h.dtype)
                h[0, -1, :].add_(delta)
                self.n_interventions += 1

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
    """Detect on one axis, correct along another.

    Trigger: fires when <h, v_detect> < tau_detect  (e.g. assistant axis).
    Correction: h <- h - v_correct * min(<h, v_correct> - tau_correct, 0)

    The detection axis gates the intervention (controls selectivity).
    The correction axis performs the actual capping (controls effectiveness).
    """

    def __init__(
        self,
        layer_module: nn.Module,
        detect_axis: torch.Tensor,
        detect_threshold: float,
        correct_axis: torch.Tensor,
        correct_threshold: float,
    ):
        self._layer = layer_module
        self._detect_axis = detect_axis.float()
        self._tau_detect = detect_threshold
        self._correct_axis = correct_axis.float()
        self._tau_correct = correct_threshold
        self._detect_device: Optional[torch.Tensor] = None
        self._correct_device: Optional[torch.Tensor] = None
        self.n_triggered = 0
        self.n_corrected = 0
        self._handle = None

    def __enter__(self):
        def hook_fn(module, input, output):
            if torch.is_tensor(output):
                h = output
            else:
                h = output[0]

            h_last = h[0, -1, :].float()

            if self._detect_device is None:
                self._detect_device = self._detect_axis.to(h.device)
                self._correct_device = self._correct_axis.to(h.device)

            detect_proj = (h_last @ self._detect_device).item()
            if detect_proj < self._tau_detect:
                self.n_triggered += 1

                correct_proj = (h_last @ self._correct_device).item()
                if correct_proj < self._tau_correct:
                    delta = (self._tau_correct - correct_proj) * self._correct_device.to(h.dtype)
                    h[0, -1, :].add_(delta)
                    self.n_corrected += 1

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

def _collect_projections(
    exp: SteeringExperiment,
    prompts: list[str],
    axis_directions: dict[str, torch.Tensor],
    cap_layers: list[int],
    max_new_tokens: int = 64,
    label: str = "",
) -> dict[str, dict[int, list[float]]]:
    """Run prompts through the model and collect per-axis, per-layer projections.

    Returns:
        projs[axis_name][layer_idx] = list[float] across all prompt/step pairs.
    """
    projs: dict[str, dict[int, list[float]]] = {
        name: {layer_idx: [] for layer_idx in cap_layers}
        for name in axis_directions
    }
    desc = f"  {label}" if label else "Collecting"
    for prompt in tqdm(prompts, desc=desc, leave=False):
        input_ids = exp.tokenize(prompt)
        with ExitStack() as stack:
            trackers: dict[tuple, _AxisProjectionTracker] = {}
            for axis_name, axis_unit in axis_directions.items():
                for layer_idx in cap_layers:
                    t = _AxisProjectionTracker(exp.layers[layer_idx], axis_unit)
                    stack.enter_context(t)
                    trackers[(axis_name, layer_idx)] = t
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                exp.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        for (axis_name, layer_idx), tracker in trackers.items():
            projs[axis_name][layer_idx].extend(tracker.projections)
    return projs


def compute_discriminative_thresholds(
    exp: SteeringExperiment,
    benign_prompts: list[str],
    jailbreak_prompts: list[str],
    axis_directions: dict[str, torch.Tensor],
    cap_layers: list[int],
    max_new_tokens: int = 64,
) -> dict[str, dict[int, dict[str, float]]]:
    """Compute per-layer thresholds from the midpoint of benign/jailbreak distributions.

    Sets tau = (mean_benign + mean_jailbreak) / 2 at each cap layer.

    Returns:
        thresholds[axis_name][layer_idx] = {
            "optimal", "mean_benign", "mean_jailbreak",
            "std_benign", "std_jailbreak", "separation"
        }
    """
    logger.info(
        "Computing discriminative thresholds at L%d-L%d "
        "(%d benign, %d jailbreak prompts)...",
        cap_layers[0], cap_layers[-1],
        len(benign_prompts), len(jailbreak_prompts),
    )

    benign_projs    = _collect_projections(exp, benign_prompts,    axis_directions, cap_layers, max_new_tokens, label="Benign")
    jailbreak_projs = _collect_projections(exp, jailbreak_prompts, axis_directions, cap_layers, max_new_tokens, label="Jailbreak")

    thresholds: dict[str, dict[int, dict[str, float]]] = {}
    for axis_name in axis_directions:
        thresholds[axis_name] = {}
        for layer_idx in cap_layers:
            b_arr = np.array(benign_projs[axis_name][layer_idx],   dtype=np.float32)
            j_arr = np.array(jailbreak_projs[axis_name][layer_idx], dtype=np.float32)
            mean_b, mean_j = float(b_arr.mean()), float(j_arr.mean())
            tau = (mean_b + mean_j) / 2.0
            separation = mean_b - mean_j
            thresholds[axis_name][layer_idx] = {
                "optimal":        tau,
                "mean_benign":    mean_b,
                "mean_jailbreak": mean_j,
                "std_benign":     float(b_arr.std()),
                "std_jailbreak":  float(j_arr.std()),
                "separation":     separation,
            }

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


# ---------------------------------------------------------------------------
# Compliance axis construction
# ---------------------------------------------------------------------------

def compute_pca_compliance_axis(
    exp: SteeringExperiment,
    refusing_prompts: list[str],
    compliant_prompts: list[str],
    cap_layers: list[int],
    axis_name: str = "pca_compliance",
) -> torch.Tensor:
    """Compute compliance direction via PCA on pooled refusing + compliant activations.

    PC1 captures the dominant axis of variation. Sign is fixed so PC1 points
    from compliant toward refusing.
    """
    ref_layer = cap_layers[-1]
    logger.info(
        "Computing %s PCA compliance axis at L%d (%d refusing, %d compliant)...",
        axis_name, ref_layer, len(refusing_prompts), len(compliant_prompts),
    )

    refusing_acts, compliant_acts = [], []

    for prompt in tqdm(refusing_prompts, desc=f"  {axis_name} refusing", leave=False):
        ids = exp.tokenize(prompt)
        acts, _ = exp.get_baseline_trajectory(ids)
        refusing_acts.append(acts[ref_layer].float())

    for prompt in tqdm(compliant_prompts, desc=f"  {axis_name} compliant", leave=False):
        ids = exp.tokenize(prompt)
        acts, _ = exp.get_baseline_trajectory(ids)
        compliant_acts.append(acts[ref_layer].float())

    refusing_stack  = torch.stack(refusing_acts)
    compliant_stack = torch.stack(compliant_acts)
    pooled = torch.cat([refusing_stack, compliant_stack], dim=0)

    pooled_centered = pooled - pooled.mean(dim=0)
    _, S, Vt = torch.linalg.svd(pooled_centered, full_matrices=False)
    pc1 = Vt[0].float()

    mean_diff = refusing_stack.mean(0) - compliant_stack.mean(0)
    if (pc1 @ mean_diff).item() < 0:
        pc1 = -pc1

    var_explained = (S[0] ** 2 / (S ** 2).sum()).item() * 100

    pc1 = pc1 / pc1.norm()
    logger.info("  %s: var_explained=%.1f%%", axis_name, var_explained)

    return pc1.cpu()


def compute_mean_diff_compliance_axis(
    exp: SteeringExperiment,
    refusing_prompts: list[str],
    compliant_prompts: list[str],
    cap_layers: list[int],
    axis_name: str = "mean_diff_compliance",
) -> torch.Tensor:
    """Compliance direction via mean difference (refusing − compliant), L2-normalised."""
    ref_layer = cap_layers[-1]
    logger.info(
        "Computing %s mean-diff compliance axis at L%d (%d refusing, %d compliant)...",
        axis_name, ref_layer, len(refusing_prompts), len(compliant_prompts),
    )

    refusing_acts, compliant_acts = [], []

    for prompt in tqdm(refusing_prompts, desc=f"  {axis_name} refusing", leave=False):
        ids = exp.tokenize(prompt)
        acts, _ = exp.get_baseline_trajectory(ids)
        refusing_acts.append(acts[ref_layer].float())

    for prompt in tqdm(compliant_prompts, desc=f"  {axis_name} compliant", leave=False):
        ids = exp.tokenize(prompt)
        acts, _ = exp.get_baseline_trajectory(ids)
        compliant_acts.append(acts[ref_layer].float())

    refusing_stack = torch.stack(refusing_acts)
    compliant_stack = torch.stack(compliant_acts)

    mean_diff = refusing_stack.mean(0) - compliant_stack.mean(0)
    mean_diff = mean_diff / mean_diff.norm()

    logger.info("  %s: axis computed (L2-normalised)", axis_name)
    return mean_diff.cpu()


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_baseline(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    axis_directions: dict[str, torch.Tensor],
    track_layers: list[int],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> tuple:
    """Generate baseline (uncapped) while tracking projections on all axes.

    Returns:
        (sequences, scores, projs) where
        projs[axis_name][layer_idx] = list[float] per decode step.
    """
    with ExitStack() as stack:
        trackers: dict[tuple, _AxisProjectionTracker] = {}
        for axis_name, axis_unit in axis_directions.items():
            for layer_idx in track_layers:
                t = _AxisProjectionTracker(exp.layers[layer_idx], axis_unit)
                stack.enter_context(t)
                trackers[(axis_name, layer_idx)] = t

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
    cap_layers: list[int],
    axis_unit: torch.Tensor,
    per_layer_thresholds: dict[int, float],
    track_layers: list[int],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> tuple:
    """Generate with capping hooks active across all cap_layers.

    Returns:
        (sequences, scores, projs, n_interventions, active_layers) where
        projs[layer_idx] = list[float] per decode step,
        n_interventions = total corrections across all cap layers,
        active_layers = list of layer indices where capping fired.
    """
    cap_hooks = [
        _CappingHook(exp.layers[layer_idx], axis_unit, per_layer_thresholds[layer_idx])
        for layer_idx in cap_layers
    ]

    with ExitStack() as stack:
        for hook in cap_hooks:
            stack.enter_context(hook)

        trackers: dict[int, _AxisProjectionTracker] = {}
        for layer_idx in track_layers:
            t = _AxisProjectionTracker(exp.layers[layer_idx], axis_unit)
            stack.enter_context(t)
            trackers[layer_idx] = t

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

    n_interventions = sum(h.n_interventions for h in cap_hooks)
    active_layers = [li for li, h in zip(cap_layers, cap_hooks) if h.n_interventions > 0]
    return output.sequences, output.scores, projs, n_interventions, active_layers


def generate_cross_capped(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    cap_layers: list[int],
    detect_axis: torch.Tensor,
    correct_axis: torch.Tensor,
    detect_thresholds: dict[int, float],
    correct_thresholds: dict[int, float],
    track_layers: list[int],
    track_axis: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> tuple:
    """Generate with cross-axis capping: detect on one axis, correct on another.

    Returns:
        (sequences, scores, projs, n_triggered, n_corrected, corrected_layers)
    """
    hooks = [
        _CrossAxisCappingHook(
            exp.layers[layer_idx],
            detect_axis, detect_thresholds[layer_idx],
            correct_axis, correct_thresholds[layer_idx],
        )
        for layer_idx in cap_layers
    ]

    with ExitStack() as stack:
        for hook in hooks:
            stack.enter_context(hook)

        trackers: dict[int, _AxisProjectionTracker] = {}
        for layer_idx in track_layers:
            t = _AxisProjectionTracker(exp.layers[layer_idx], track_axis)
            stack.enter_context(t)
            trackers[layer_idx] = t

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

    n_triggered = sum(h.n_triggered for h in hooks)
    n_corrected = sum(h.n_corrected for h in hooks)
    corrected_layers = [li for li, h in zip(cap_layers, hooks) if h.n_corrected > 0]
    return output.sequences, output.scores, projs, n_triggered, n_corrected, corrected_layers
