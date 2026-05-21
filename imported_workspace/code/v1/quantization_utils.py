"""
Advanced Quantization Strategies for CLIP
==========================================

Includes:
- GPTQ (data-free weight quantization)
- AWQ (Activation-aware Weight Quantization)  
- Mixed-precision quantization
- Outlier-aware quantization
- Per-layer sensitivity analysis
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Quantization Utilities
# =============================================================================

def compute_scale_zero_point(tensor: torch.Tensor, bits: int, 
                              symmetric: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute quantization scale and zero point."""
    if symmetric:
        max_val = tensor.abs().max()
        qmin, qmax = -(2**(bits-1)), 2**(bits-1) - 1
        scale = max_val / qmax
        zero_point = torch.zeros_like(scale)
    else:
        min_val, max_val = tensor.min(), tensor.max()
        qmin, qmax = 0, 2**bits - 1
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
    
    return scale, zero_point


def fake_quantize(tensor: torch.Tensor, bits: int, 
                  symmetric: bool = False) -> torch.Tensor:
    """Simulate quantization without actually converting dtype."""
    scale, zero_point = compute_scale_zero_point(tensor, bits, symmetric)
    qmin, qmax = 0, 2**bits - 1
    
    # Quantize then dequantize
    q = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
    return (q - zero_point) * scale


def compute_quantization_error(original: torch.Tensor, quantized: torch.Tensor) -> Dict[str, float]:
    """Compute various quantization error metrics."""
    diff = original - quantized
    return {
        "mse": (diff ** 2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_error": diff.abs().max().item(),
        "relative_error": (diff.abs() / (original.abs() + 1e-8)).mean().item(),
        "snr_db": 10 * torch.log10((original ** 2).mean() / ((diff ** 2).mean() + 1e-8)).item()
    }


# =============================================================================
# Per-Layer Sensitivity Analysis
# =============================================================================

class LayerSensitivityAnalyzer:
    """Analyze sensitivity of each layer to quantization."""
    
    def __init__(self, model: nn.Module, bits_range: List[int] = [2, 3, 4, 6, 8]):
        self.model = model
        self.bits_range = bits_range
        self.sensitivity_scores: Dict[str, Dict[int, float]] = {}
        
    def compute_weight_sensitivity(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Compute sensitivity based on weight quantization error."""
        results = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                results[name] = {}
                
                for bits in self.bits_range:
                    quantized = fake_quantize(weight, bits)
                    results[name][bits] = compute_quantization_error(weight, quantized)
        
        return results
    
    def compute_activation_sensitivity(self, 
                                       calibration_data: List[torch.Tensor]) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Compute sensitivity based on activation quantization error."""
        activation_cache = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if name not in activation_cache:
                    activation_cache[name] = []
                activation_cache[name].append(output.detach())
            return hook
        
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Run calibration data
        self.model.eval()
        with torch.no_grad():
            for data in calibration_data:
                self.model(data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute sensitivity
        results = {}
        for name, activations in activation_cache.items():
            all_activations = torch.cat(activations, dim=0)
            results[name] = {}
            
            for bits in self.bits_range:
                quantized = fake_quantize(all_activations, bits)
                results[name][bits] = compute_quantization_error(all_activations, quantized)
        
        return results
    
    def get_layer_importance_ranking(self, metric: str = "snr_db") -> List[Tuple[str, float]]:
        """Rank layers by their sensitivity to quantization."""
        if not self.sensitivity_scores:
            self.sensitivity_scores = self.compute_weight_sensitivity()
        
        # Use the difference between 8-bit and 4-bit as importance measure
        importance = {}
        for layer_name, bit_results in self.sensitivity_scores.items():
            if 8 in bit_results and 4 in bit_results:
                importance[layer_name] = bit_results[4][metric] - bit_results[8][metric]
        
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)


# =============================================================================
# GPTQ-style Quantization
# =============================================================================

class GPTQQuantizer:
    """
    GPTQ-style quantization for linear layers.
    Uses Hessian-based weight update for minimal accuracy loss.
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128, 
                 actorder: bool = True, percdamp: float = 0.01):
        self.bits = bits
        self.group_size = group_size
        self.actorder = actorder
        self.percdamp = percdamp
        
    def quantize_layer(self, layer: nn.Linear, 
                       calibration_data: List[torch.Tensor]) -> nn.Linear:
        """Quantize a single linear layer using GPTQ."""
        W = layer.weight.data.clone()
        nsamples = len(calibration_data)
        
        # Compute Hessian approximation
        H = torch.zeros((W.shape[1], W.shape[1]), device=W.device)
        for inp in calibration_data:
            inp = inp.reshape(-1, inp.shape[-1])
            H += inp.T @ inp
        H /= nsamples
        
        # Add damping
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        damp = self.percdamp * torch.mean(torch.diag(H))
        H += damp * torch.eye(H.shape[0], device=H.device)
        
        # Cholesky decomposition
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
        
        # Quantize weights column by column
        Q = torch.zeros_like(W)
        
        for i in range(W.shape[1]):
            w = W[:, i]
            d = H_inv[i, i]
            
            # Quantize
            q = self._quantize_weight(w)
            Q[:, i] = q
            
            # Update remaining weights (error compensation)
            err = (w - q) / d
            W[:, i+1:] -= err.unsqueeze(1) * H_inv[i, i+1:].unsqueeze(0)
        
        # Create new layer with quantized weights
        new_layer = nn.Linear(layer.in_features, layer.out_features, 
                             bias=layer.bias is not None)
        new_layer.weight.data = Q
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data.clone()
        
        return new_layer
    
    def _quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize a weight vector."""
        return fake_quantize(weight, self.bits, symmetric=True)
    
    def quantize_model(self, model: nn.Module, 
                       calibration_data: List[torch.Tensor]) -> nn.Module:
        """Quantize entire model using GPTQ."""
        # Collect activations for each layer
        activation_cache = defaultdict(list)
        
        def make_hook(name):
            def hook(module, input, output):
                activation_cache[name].append(input[0].detach())
            return hook
        
        # Register hooks
        hooks = []
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))
                layer_names.append(name)
        
        # Run calibration
        model.eval()
        with torch.no_grad():
            for data in calibration_data:
                model(data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Quantize each layer
        for name in layer_names:
            # Get module
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            layer = getattr(module, parts[-1])
            
            # Quantize
            quantized_layer = self.quantize_layer(layer, activation_cache[name])
            setattr(module, parts[-1], quantized_layer)
        
        return model


# =============================================================================
# AWQ-style Quantization  
# =============================================================================

class AWQQuantizer:
    """
    Activation-aware Weight Quantization.
    Scales weights based on activation magnitudes before quantization.
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        
    def compute_scale(self, weight: torch.Tensor, 
                     activations: torch.Tensor) -> torch.Tensor:
        """Compute optimal scaling factors."""
        # Compute activation magnitudes per channel
        act_scales = activations.abs().mean(dim=0)
        
        # Search for optimal scaling
        best_scale = torch.ones_like(act_scales)
        best_error = float('inf')
        
        for scale_factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            scale = act_scales.pow(scale_factor)
            scale = scale / scale.mean()
            
            # Scale weights
            scaled_weight = weight * scale.unsqueeze(0)
            
            # Quantize
            quantized = fake_quantize(scaled_weight, self.bits)
            
            # Unscale
            unscaled = quantized / scale.unsqueeze(0)
            
            # Compute error
            error = ((weight - unscaled) ** 2).mean().item()
            
            if error < best_error:
                best_error = error
                best_scale = scale.clone()
        
        return best_scale
    
    def quantize_layer(self, layer: nn.Linear, 
                       activations: torch.Tensor) -> Tuple[nn.Linear, torch.Tensor]:
        """Quantize a single linear layer using AWQ."""
        weight = layer.weight.data
        
        # Compute scaling
        scale = self.compute_scale(weight, activations)
        
        # Scale and quantize
        scaled_weight = weight * scale.unsqueeze(0)
        quantized = fake_quantize(scaled_weight, self.bits)
        
        # Store scale for inference (to unscale activations)
        new_layer = nn.Linear(layer.in_features, layer.out_features,
                             bias=layer.bias is not None)
        new_layer.weight.data = quantized
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data.clone()
        
        return new_layer, scale


# =============================================================================
# Mixed Precision Quantization
# =============================================================================

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed-precision quantization."""
    default_bits: int = 8
    layer_bits: Dict[str, int] = None
    sensitive_layers_bits: int = 8  # Keep sensitive layers at higher precision
    sensitivity_threshold: float = 0.1  # SNR degradation threshold
    
    def __post_init__(self):
        if self.layer_bits is None:
            self.layer_bits = {}


class MixedPrecisionQuantizer:
    """
    Apply different bit widths to different layers based on sensitivity.
    """
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        
    def auto_configure(self, model: nn.Module, 
                       target_compression: float = 4.0) -> MixedPrecisionConfig:
        """Automatically configure bit widths to achieve target compression."""
        analyzer = LayerSensitivityAnalyzer(model)
        sensitivity = analyzer.compute_weight_sensitivity()
        ranking = analyzer.get_layer_importance_ranking()
        
        # Count total weights
        total_params = sum(p.numel() for n, p in model.named_parameters() 
                          if any(n.startswith(ln) for ln, _ in ranking))
        
        # Assign bits greedily
        layer_bits = {}
        current_bits_total = total_params * 32  # Start with FP32
        
        for layer_name, importance in ranking:
            layer_params = 0
            for n, p in model.named_parameters():
                if n.startswith(layer_name):
                    layer_params += p.numel()
            
            # Decide bits based on importance
            if importance < self.config.sensitivity_threshold:
                bits = 4
            else:
                bits = self.config.sensitive_layers_bits
            
            layer_bits[layer_name] = bits
            current_bits_total -= layer_params * 32
            current_bits_total += layer_params * bits
        
        self.config.layer_bits = layer_bits
        return self.config
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply mixed-precision quantization."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                bits = self.config.layer_bits.get(name, self.config.default_bits)
                
                # Apply fake quantization for simulation
                with torch.no_grad():
                    module.weight.data = fake_quantize(module.weight.data, bits)
                    
        return model


# =============================================================================
# Outlier-Aware Quantization
# =============================================================================

class OutlierAwareQuantizer:
    """
    Handle weight/activation outliers that hurt quantization.
    Options: clip, absorb into scale, or keep in FP16.
    """
    
    def __init__(self, bits: int = 4, outlier_threshold: float = 3.0,
                 outlier_handling: str = "clip"):  # "clip", "absorb", "fp16"
        self.bits = bits
        self.outlier_threshold = outlier_threshold
        self.outlier_handling = outlier_handling
        
    def detect_outliers(self, tensor: torch.Tensor) -> torch.Tensor:
        """Detect outlier values in tensor."""
        mean = tensor.mean()
        std = tensor.std()
        threshold = self.outlier_threshold * std
        
        outliers = torch.abs(tensor - mean) > threshold
        return outliers
    
    def quantize_with_outlier_handling(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Quantize weight tensor with outlier handling."""
        outlier_mask = self.detect_outliers(weight)
        
        if self.outlier_handling == "clip":
            # Clip outliers before quantization
            mean = weight.mean()
            std = weight.std()
            clipped = torch.clamp(weight, 
                                 mean - self.outlier_threshold * std,
                                 mean + self.outlier_threshold * std)
            quantized = fake_quantize(clipped, self.bits)
            return quantized, None
            
        elif self.outlier_handling == "absorb":
            # Absorb outliers into quantization scale
            max_val = weight.abs().max()
            # Use larger scale to accommodate outliers
            quantized = fake_quantize(weight, self.bits, symmetric=True)
            return quantized, None
            
        elif self.outlier_handling == "fp16":
            # Keep outliers in FP16, quantize the rest
            outlier_values = weight[outlier_mask].clone()
            weight_copy = weight.clone()
            weight_copy[outlier_mask] = 0  # Zero out outliers
            quantized = fake_quantize(weight_copy, self.bits)
            # Return both quantized and outlier tensor
            return quantized, (outlier_mask, outlier_values)
        
        else:
            raise ValueError(f"Unknown outlier handling: {self.outlier_handling}")


# =============================================================================
# Quantization-Aware Training (QAT) Support
# =============================================================================

class QuantizationAwareTraining:
    """
    Support for quantization-aware fine-tuning.
    Uses straight-through estimator (STE) for gradients.
    """
    
    def __init__(self, bits: int = 8, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Replace layers with QAT-enabled versions."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Wrap with fake quantization
                qat_module = QATLinear(module, self.bits, self.symmetric)
                
                # Replace in parent
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], qat_module)
        
        return model


class QATLinear(nn.Module):
    """Linear layer with fake quantization for QAT."""
    
    def __init__(self, linear: nn.Linear, bits: int = 8, symmetric: bool = True):
        super().__init__()
        self.linear = linear
        self.bits = bits
        self.symmetric = symmetric
        
        # Learnable scale and zero point
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fake quantize weights
        weight = self.linear.weight
        q_weight = self._fake_quantize(weight)
        
        # Forward pass
        output = nn.functional.linear(x, q_weight, self.linear.bias)
        
        return output
    
    def _fake_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fake quantization with straight-through estimator."""
        # Compute scale
        if self.symmetric:
            max_val = tensor.abs().max()
            qmax = 2**(self.bits - 1) - 1
            scale = max_val / qmax
        else:
            min_val, max_val = tensor.min(), tensor.max()
            qmax = 2**self.bits - 1
            scale = (max_val - min_val) / qmax
        
        # Quantize
        if self.symmetric:
            q = torch.round(tensor / scale)
            q = torch.clamp(q, -qmax, qmax)
            dequant = q * scale
        else:
            zero_point = -min_val / scale
            q = torch.round(tensor / scale + zero_point)
            q = torch.clamp(q, 0, qmax)
            dequant = (q - zero_point) * scale
        
        # Straight-through estimator
        return tensor + (dequant - tensor).detach()


# =============================================================================
# Utility Functions
# =============================================================================

def get_model_size(model: nn.Module, bits_per_param: int = 32) -> int:
    """Calculate model size in bytes."""
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * bits_per_param) // 8


def estimate_memory_savings(model: nn.Module, target_bits: int) -> Dict[str, float]:
    """Estimate memory savings from quantization."""
    fp32_size = get_model_size(model, 32)
    quantized_size = get_model_size(model, target_bits)
    
    return {
        "original_size_mb": fp32_size / (1024 * 1024),
        "quantized_size_mb": quantized_size / (1024 * 1024),
        "compression_ratio": fp32_size / quantized_size,
        "memory_savings_percent": (1 - quantized_size / fp32_size) * 100
    }


def benchmark_inference_speed(model: nn.Module, input_shape: Tuple[int, ...],
                              num_iterations: int = 100, 
                              warmup: int = 10) -> Dict[str, float]:
    """Benchmark model inference speed."""
    import time
    
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return {
        "total_time_seconds": elapsed,
        "avg_time_ms": (elapsed / num_iterations) * 1000,
        "throughput_samples_per_sec": num_iterations / elapsed
    }
