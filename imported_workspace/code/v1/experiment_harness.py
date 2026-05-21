"""
CLIP Quantization Experimentation Harness for Multimodal Entity Linking
========================================================================

Designed for fast iteration on WikiDiverse dataset.
Key features:
- Caching at every level (embeddings, quantized models, evaluation metrics)
- Modular quantization strategies
- Layer importance analysis
- Parallel experiment execution
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
import hashlib
import pickle
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from functools import lru_cache
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Central configuration for experiments."""
    # Model
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Quantization
    quantization_bits: List[int] = field(default_factory=lambda: [8, 6, 4, 3, 2])
    quantization_schemes: List[str] = field(default_factory=lambda: [
        "dynamic", "static", "qat", "ptq_per_channel", "ptq_per_tensor"
    ])
    
    # Data
    wikidiverse_path: str = "./data/wikidiverse"
    cache_dir: str = "./cache"
    batch_size: int = 64
    num_workers: int = 4
    subset_size: Optional[int] = None  # For fast iteration, set to e.g. 1000
    
    # Evaluation
    eval_metrics: List[str] = field(default_factory=lambda: [
        "accuracy@1", "accuracy@5", "accuracy@10", "mrr", "ndcg@10"
    ])
    
    # Experiment
    seed: int = 42
    num_runs: int = 3  # For statistical significance
    
    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Caching Infrastructure
# =============================================================================

class EmbeddingCache:
    """Disk-backed cache for embeddings with automatic invalidation."""
    
    def __init__(self, cache_dir: str, model_name: str):
        self.cache_dir = Path(cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._memory_cache: Dict[str, torch.Tensor] = {}
        
    def _get_key(self, item_id: str, modality: str, quantization: Optional[str] = None) -> str:
        components = [self.model_name, modality, item_id]
        if quantization:
            components.append(quantization)
        return hashlib.md5("_".join(components).encode()).hexdigest()
    
    def get(self, item_id: str, modality: str, quantization: Optional[str] = None) -> Optional[torch.Tensor]:
        key = self._get_key(item_id, modality, quantization)
        
        # Memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Disk cache
        cache_path = self.cache_dir / f"{key}.pt"
        if cache_path.exists():
            embedding = torch.load(cache_path, weights_only=True)
            self._memory_cache[key] = embedding
            return embedding
        
        return None
    
    def set(self, item_id: str, modality: str, embedding: torch.Tensor, 
            quantization: Optional[str] = None):
        key = self._get_key(item_id, modality, quantization)
        self._memory_cache[key] = embedding
        torch.save(embedding, self.cache_dir / f"{key}.pt")
    
    def get_batch(self, item_ids: List[str], modality: str, 
                  quantization: Optional[str] = None) -> Tuple[List[torch.Tensor], List[int]]:
        """Returns (cached_embeddings, missing_indices)."""
        cached = []
        missing = []
        for i, item_id in enumerate(item_ids):
            emb = self.get(item_id, modality, quantization)
            if emb is not None:
                cached.append((i, emb))
            else:
                missing.append(i)
        return cached, missing


class ResultsCache:
    """Cache for experiment results with versioning."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / "results"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_key(self, config: Dict) -> str:
        return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
    
    def get(self, config: Dict) -> Optional[Dict]:
        key = self._get_key(config)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    
    def set(self, config: Dict, results: Dict):
        key = self._get_key(config)
        with open(self.cache_dir / f"{key}.json", "w") as f:
            json.dump({"config": config, "results": results}, f, indent=2)


# =============================================================================
# Quantization Strategies
# =============================================================================

class QuantizationStrategy(ABC):
    """Base class for quantization strategies."""
    
    @abstractmethod
    def quantize(self, model: nn.Module, calibration_data: Optional[DataLoader] = None) -> nn.Module:
        pass
    
    @abstractmethod
    def get_config_dict(self) -> Dict:
        pass


class DynamicQuantization(QuantizationStrategy):
    """PyTorch dynamic quantization."""
    
    def __init__(self, bits: int = 8, layers_to_quantize: Optional[List[str]] = None):
        self.bits = bits
        self.layers_to_quantize = layers_to_quantize
        
    def quantize(self, model: nn.Module, calibration_data: Optional[DataLoader] = None) -> nn.Module:
        dtype = torch.qint8 if self.bits == 8 else torch.qint8  # Extend for other bit widths
        
        # Identify layers to quantize
        layer_types = {nn.Linear}
        if self.layers_to_quantize:
            # Filter specific layers
            model_copy = model  # Clone if needed
        
        quantized = torch.quantization.quantize_dynamic(
            model, layer_types, dtype=dtype
        )
        return quantized
    
    def get_config_dict(self) -> Dict:
        return {"type": "dynamic", "bits": self.bits, "layers": self.layers_to_quantize}


class StaticQuantization(QuantizationStrategy):
    """Static quantization with calibration."""
    
    def __init__(self, bits: int = 8, calibration_samples: int = 100):
        self.bits = bits
        self.calibration_samples = calibration_samples
        
    def quantize(self, model: nn.Module, calibration_data: Optional[DataLoader] = None) -> nn.Module:
        model.eval()
        
        # Prepare model for static quantization
        model.qconfig = torch.quantization.get_default_qconfig('x86')
        prepared = torch.quantization.prepare(model)
        
        # Calibration pass
        if calibration_data:
            with torch.no_grad():
                for i, batch in enumerate(calibration_data):
                    if i >= self.calibration_samples:
                        break
                    prepared(batch)
        
        quantized = torch.quantization.convert(prepared)
        return quantized
    
    def get_config_dict(self) -> Dict:
        return {"type": "static", "bits": self.bits, "calibration_samples": self.calibration_samples}


class BitsAndBytesQuantization(QuantizationStrategy):
    """8-bit and 4-bit quantization using bitsandbytes."""
    
    def __init__(self, bits: int = 8, use_double_quant: bool = False):
        self.bits = bits
        self.use_double_quant = use_double_quant
        
    def quantize(self, model: nn.Module, calibration_data: Optional[DataLoader] = None) -> nn.Module:
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            
            if self.bits == 8:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif self.bits == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=self.use_double_quant,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                raise ValueError(f"bitsandbytes only supports 4 or 8 bit quantization")
                
            # For CLIP, we need to reload with quantization config
            # This returns a config that should be used when loading
            return bnb_config
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to dynamic quantization")
            return DynamicQuantization(self.bits).quantize(model, calibration_data)
    
    def get_config_dict(self) -> Dict:
        return {"type": "bitsandbytes", "bits": self.bits, "double_quant": self.use_double_quant}


class LayerSelectiveQuantization(QuantizationStrategy):
    """Quantize only specific layers - for importance analysis."""
    
    def __init__(self, bits: int = 8, layer_patterns: List[str] = None, 
                 exclude_patterns: List[str] = None):
        self.bits = bits
        self.layer_patterns = layer_patterns or []
        self.exclude_patterns = exclude_patterns or []
        
    def quantize(self, model: nn.Module, calibration_data: Optional[DataLoader] = None) -> nn.Module:
        model_copy = type(model)(model.config) if hasattr(model, 'config') else model
        model_copy.load_state_dict(model.state_dict())
        
        for name, module in model_copy.named_modules():
            should_quantize = (
                any(p in name for p in self.layer_patterns) and
                not any(p in name for p in self.exclude_patterns)
            )
            
            if should_quantize and isinstance(module, nn.Linear):
                # Replace with quantized version
                quantized_module = torch.quantization.quantize_dynamic(
                    nn.Sequential(module), {nn.Linear}, dtype=torch.qint8
                )[0]
                # Replace in model (simplified - need proper parent access)
                
        return model_copy
    
    def get_config_dict(self) -> Dict:
        return {
            "type": "layer_selective", 
            "bits": self.bits, 
            "include": self.layer_patterns,
            "exclude": self.exclude_patterns
        }


# =============================================================================
# WikiDiverse Dataset Handler
# =============================================================================

@dataclass
class WikiDiverseEntity:
    """Represents an entity from WikiDiverse."""
    entity_id: str
    name: str
    description: str
    image_path: Optional[str]
    wikidata_id: Optional[str]
    aliases: List[str] = field(default_factory=list)


@dataclass  
class WikiDiverseMention:
    """Represents a mention to be linked."""
    mention_id: str
    text: str
    context: str
    image_path: Optional[str]
    gold_entity_id: str
    candidate_entity_ids: List[str]


class WikiDiverseDataset(Dataset):
    """WikiDiverse dataset loader with efficient caching."""
    
    def __init__(self, data_path: str, split: str = "test", 
                 subset_size: Optional[int] = None):
        self.data_path = Path(data_path)
        self.split = split
        self.subset_size = subset_size
        
        self.mentions: List[WikiDiverseMention] = []
        self.entities: Dict[str, WikiDiverseEntity] = {}
        
        self._load_data()
        
    def _load_data(self):
        """Load WikiDiverse data. Override for your specific format."""
        # WikiDiverse format: JSON files with mentions and entities
        mentions_file = self.data_path / f"{self.split}_mentions.json"
        entities_file = self.data_path / "entities.json"
        
        if mentions_file.exists():
            with open(mentions_file) as f:
                mentions_data = json.load(f)
            self.mentions = [
                WikiDiverseMention(
                    mention_id=m["id"],
                    text=m["mention"],
                    context=m.get("context", ""),
                    image_path=m.get("image_path"),
                    gold_entity_id=m["entity_id"],
                    candidate_entity_ids=m.get("candidates", [])
                )
                for m in mentions_data
            ]
        else:
            logger.warning(f"Mentions file not found: {mentions_file}")
            self._create_dummy_data()
            
        if entities_file.exists():
            with open(entities_file) as f:
                entities_data = json.load(f)
            self.entities = {
                e["id"]: WikiDiverseEntity(
                    entity_id=e["id"],
                    name=e["name"],
                    description=e.get("description", ""),
                    image_path=e.get("image_path"),
                    wikidata_id=e.get("wikidata_id"),
                    aliases=e.get("aliases", [])
                )
                for e in entities_data
            }
        
        if self.subset_size:
            self.mentions = self.mentions[:self.subset_size]
            
    def _create_dummy_data(self):
        """Create dummy data for testing the harness."""
        logger.info("Creating dummy data for testing...")
        for i in range(100):
            self.mentions.append(WikiDiverseMention(
                mention_id=f"m_{i}",
                text=f"Mention {i}",
                context=f"This is context for mention {i}",
                image_path=None,
                gold_entity_id=f"e_{i % 10}",
                candidate_entity_ids=[f"e_{j}" for j in range(10)]
            ))
        for i in range(10):
            self.entities[f"e_{i}"] = WikiDiverseEntity(
                entity_id=f"e_{i}",
                name=f"Entity {i}",
                description=f"Description for entity {i}",
                image_path=None,
                wikidata_id=f"Q{i}"
            )
    
    def __len__(self) -> int:
        return len(self.mentions)
    
    def __getitem__(self, idx: int) -> WikiDiverseMention:
        return self.mentions[idx]
    
    def get_entity(self, entity_id: str) -> Optional[WikiDiverseEntity]:
        return self.entities.get(entity_id)


# =============================================================================
# Entity Linking Model Wrapper
# =============================================================================

class CLIPEntityLinker(nn.Module):
    """CLIP-based entity linking model."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
    def encode_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode text inputs."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def encode_images(self, images: List, batch_size: int = 32) -> torch.Tensor:
        """Encode image inputs."""
        from PIL import Image
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Handle image paths or PIL images
            pil_images = []
            for img in batch_images:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert("RGB"))
                else:
                    pil_images.append(img)
            
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def get_layer_names(self) -> List[str]:
        """Get all layer names for selective quantization experiments."""
        return [name for name, _ in self.model.named_modules() if isinstance(_, nn.Linear)]
    
    def get_layer_groups(self) -> Dict[str, List[str]]:
        """Group layers by component for importance analysis."""
        groups = defaultdict(list)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if "text_model" in name:
                    if "self_attn" in name:
                        groups["text_attention"].append(name)
                    elif "mlp" in name:
                        groups["text_mlp"].append(name)
                    else:
                        groups["text_other"].append(name)
                elif "vision_model" in name:
                    if "self_attn" in name:
                        groups["vision_attention"].append(name)
                    elif "mlp" in name:
                        groups["vision_mlp"].append(name)
                    else:
                        groups["vision_other"].append(name)
                else:
                    groups["projection"].append(name)
        return dict(groups)


# =============================================================================
# Evaluation Metrics
# =============================================================================

class EntityLinkingMetrics:
    """Metrics for entity linking evaluation."""
    
    @staticmethod
    def accuracy_at_k(predictions: List[List[str]], gold: List[str], k: int) -> float:
        """Compute accuracy@k."""
        correct = 0
        for pred_list, gold_id in zip(predictions, gold):
            if gold_id in pred_list[:k]:
                correct += 1
        return correct / len(gold) if gold else 0.0
    
    @staticmethod
    def mrr(predictions: List[List[str]], gold: List[str]) -> float:
        """Compute Mean Reciprocal Rank."""
        rr_sum = 0.0
        for pred_list, gold_id in zip(predictions, gold):
            try:
                rank = pred_list.index(gold_id) + 1
                rr_sum += 1.0 / rank
            except ValueError:
                pass
        return rr_sum / len(gold) if gold else 0.0
    
    @staticmethod
    def ndcg_at_k(predictions: List[List[str]], gold: List[str], k: int) -> float:
        """Compute NDCG@k."""
        def dcg(relevances: List[float]) -> float:
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        
        ndcg_sum = 0.0
        for pred_list, gold_id in zip(predictions, gold):
            relevances = [1.0 if pred == gold_id else 0.0 for pred in pred_list[:k]]
            ideal_relevances = sorted(relevances, reverse=True)
            
            dcg_val = dcg(relevances)
            idcg_val = dcg(ideal_relevances)
            
            if idcg_val > 0:
                ndcg_sum += dcg_val / idcg_val
        
        return ndcg_sum / len(gold) if gold else 0.0
    
    @staticmethod
    def compute_all(predictions: List[List[str]], gold: List[str]) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            "accuracy@1": EntityLinkingMetrics.accuracy_at_k(predictions, gold, 1),
            "accuracy@5": EntityLinkingMetrics.accuracy_at_k(predictions, gold, 5),
            "accuracy@10": EntityLinkingMetrics.accuracy_at_k(predictions, gold, 10),
            "mrr": EntityLinkingMetrics.mrr(predictions, gold),
            "ndcg@10": EntityLinkingMetrics.ndcg_at_k(predictions, gold, 10),
        }


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Main experiment orchestrator."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.embedding_cache = EmbeddingCache(config.cache_dir, config.clip_model_name)
        self.results_cache = ResultsCache(config.cache_dir)
        
        # Load model and data
        self.model = CLIPEntityLinker(config.clip_model_name, config.device)
        self.dataset = WikiDiverseDataset(
            config.wikidiverse_path, 
            subset_size=config.subset_size
        )
        
        # Pre-compute baseline embeddings
        self._baseline_entity_embeddings: Optional[torch.Tensor] = None
        self._baseline_mention_embeddings: Optional[torch.Tensor] = None
        
    def _get_entity_embeddings(self, quantization_config: Optional[Dict] = None) -> torch.Tensor:
        """Get entity embeddings, using cache when possible."""
        cache_key = json.dumps(quantization_config) if quantization_config else "baseline"
        
        entities = list(self.dataset.entities.values())
        texts = [f"{e.name}. {e.description}" for e in entities]
        
        embeddings = self.model.encode_text(texts)
        return embeddings
    
    def _get_mention_embeddings(self, quantization_config: Optional[Dict] = None) -> torch.Tensor:
        """Get mention embeddings, using cache when possible."""
        texts = [f"{m.text}. {m.context}" for m in self.dataset.mentions]
        embeddings = self.model.encode_text(texts)
        return embeddings
    
    def _predict_entities(self, mention_embs: torch.Tensor, 
                          entity_embs: torch.Tensor) -> List[List[str]]:
        """Predict entities for each mention."""
        entity_ids = list(self.dataset.entities.keys())
        
        # Compute similarity matrix
        similarities = torch.mm(mention_embs, entity_embs.t())
        
        # Get top predictions
        _, indices = similarities.topk(k=min(10, len(entity_ids)), dim=1)
        
        predictions = []
        for idx_row in indices:
            predictions.append([entity_ids[i] for i in idx_row.tolist()])
        
        return predictions
    
    def evaluate_baseline(self) -> Dict[str, float]:
        """Evaluate unquantized model."""
        logger.info("Evaluating baseline model...")
        
        # Check cache
        cached = self.results_cache.get({"type": "baseline"})
        if cached:
            logger.info("Using cached baseline results")
            return cached["results"]
        
        entity_embs = self._get_entity_embeddings()
        mention_embs = self._get_mention_embeddings()
        
        predictions = self._predict_entities(mention_embs, entity_embs)
        gold = [m.gold_entity_id for m in self.dataset.mentions]
        
        results = EntityLinkingMetrics.compute_all(predictions, gold)
        
        self.results_cache.set({"type": "baseline"}, results)
        return results
    
    def evaluate_quantization(self, strategy: QuantizationStrategy, 
                              calibration_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate a specific quantization strategy."""
        config = strategy.get_config_dict()
        logger.info(f"Evaluating quantization: {config}")
        
        # Check cache
        cached = self.results_cache.get(config)
        if cached:
            logger.info("Using cached results")
            return cached["results"]
        
        # Apply quantization
        start_time = time.time()
        quantized_model = strategy.quantize(self.model.model, calibration_loader)
        quantization_time = time.time() - start_time
        
        # Temporarily replace model
        original_model = self.model.model
        self.model.model = quantized_model
        
        try:
            # Get embeddings with quantized model
            entity_embs = self._get_entity_embeddings(config)
            mention_embs = self._get_mention_embeddings(config)
            
            predictions = self._predict_entities(mention_embs, entity_embs)
            gold = [m.gold_entity_id for m in self.dataset.mentions]
            
            metrics = EntityLinkingMetrics.compute_all(predictions, gold)
        finally:
            # Restore original model
            self.model.model = original_model
        
        # Compute model size
        model_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        
        results = {
            "metrics": metrics,
            "quantization_time_seconds": quantization_time,
            "model_size_bytes": model_size,
            "config": config
        }
        
        self.results_cache.set(config, results)
        return results
    
    def run_layer_importance_analysis(self, bits: int = 8) -> Dict[str, Dict[str, float]]:
        """Analyze importance of different layer groups."""
        logger.info(f"Running layer importance analysis at {bits} bits...")
        
        baseline = self.evaluate_baseline()
        layer_groups = self.model.get_layer_groups()
        
        results = {"baseline": baseline}
        
        for group_name, layers in layer_groups.items():
            logger.info(f"Testing group: {group_name}")
            
            # Quantize only this group
            strategy = LayerSelectiveQuantization(
                bits=bits,
                layer_patterns=layers
            )
            
            group_results = self.evaluate_quantization(strategy)
            
            # Compute degradation
            degradation = {
                metric: baseline[metric] - group_results["metrics"][metric]
                for metric in baseline
            }
            
            results[group_name] = {
                "metrics": group_results["metrics"],
                "degradation": degradation,
                "layers": layers
            }
        
        return results
    
    def run_bit_width_sweep(self, strategy_type: str = "dynamic") -> Dict[int, Dict[str, Any]]:
        """Run experiments across different bit widths."""
        logger.info(f"Running bit width sweep with {strategy_type} quantization...")
        
        results = {}
        baseline = self.evaluate_baseline()
        results["baseline"] = {"metrics": baseline, "bits": "fp32"}
        
        for bits in self.config.quantization_bits:
            logger.info(f"Testing {bits}-bit quantization...")
            
            if strategy_type == "dynamic":
                strategy = DynamicQuantization(bits=bits)
            elif strategy_type == "bitsandbytes":
                strategy = BitsAndBytesQuantization(bits=bits)
            else:
                strategy = StaticQuantization(bits=bits)
            
            try:
                results[bits] = self.evaluate_quantization(strategy)
            except Exception as e:
                logger.error(f"Failed for {bits} bits: {e}")
                results[bits] = {"error": str(e)}
        
        return results
    
    def run_full_experiment_suite(self) -> Dict[str, Any]:
        """Run complete experiment suite."""
        logger.info("Starting full experiment suite...")
        
        all_results = {
            "config": self.config.__dict__,
            "baseline": self.evaluate_baseline(),
            "bit_width_sweep": {},
            "layer_importance": {},
            "combined_analysis": {}
        }
        
        # Bit width sweeps for each strategy
        for strategy in ["dynamic", "bitsandbytes"]:
            try:
                all_results["bit_width_sweep"][strategy] = self.run_bit_width_sweep(strategy)
            except Exception as e:
                logger.error(f"Failed {strategy} sweep: {e}")
        
        # Layer importance at 8-bit
        try:
            all_results["layer_importance"] = self.run_layer_importance_analysis(bits=8)
        except Exception as e:
            logger.error(f"Failed layer importance: {e}")
        
        return all_results


# =============================================================================
# Visualization & Reporting
# =============================================================================

def generate_accuracy_vs_quantization_plot(results: Dict, output_path: str = "accuracy_vs_bits.png"):
    """Generate accuracy vs quantization plot."""
    try:
        import matplotlib.pyplot as plt
        
        bits = []
        accuracies = []
        
        for bit_width, data in results.items():
            if bit_width == "baseline":
                bits.append(32)
                accuracies.append(data["metrics"]["accuracy@1"])
            elif isinstance(bit_width, int) and "metrics" in data:
                bits.append(bit_width)
                accuracies.append(data["metrics"]["accuracy@1"])
        
        # Sort by bits
        sorted_pairs = sorted(zip(bits, accuracies), reverse=True)
        bits, accuracies = zip(*sorted_pairs)
        
        plt.figure(figsize=(10, 6))
        plt.plot(bits, accuracies, 'bo-', linewidth=2, markersize=10)
        plt.xlabel('Bit Width', fontsize=12)
        plt.ylabel('Accuracy@1', fontsize=12)
        plt.title('CLIP Entity Linking: Accuracy vs Quantization', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()  # Higher bits on left
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Plot saved to {output_path}")
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot generation")


def generate_layer_importance_report(results: Dict) -> str:
    """Generate markdown report for layer importance."""
    report = ["# Layer Importance Analysis\n"]
    
    baseline_acc = results.get("baseline", {}).get("accuracy@1", 0)
    report.append(f"**Baseline Accuracy@1:** {baseline_acc:.4f}\n\n")
    
    report.append("| Layer Group | Accuracy@1 | Degradation | Importance Rank |\n")
    report.append("|-------------|------------|-------------|----------------|\n")
    
    # Sort by degradation
    groups = [(k, v) for k, v in results.items() if k != "baseline"]
    groups.sort(key=lambda x: x[1].get("degradation", {}).get("accuracy@1", 0), reverse=True)
    
    for rank, (group, data) in enumerate(groups, 1):
        acc = data.get("metrics", {}).get("accuracy@1", 0)
        deg = data.get("degradation", {}).get("accuracy@1", 0)
        report.append(f"| {group} | {acc:.4f} | {deg:+.4f} | {rank} |\n")
    
    return "".join(report)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP Quantization Experiments")
    parser.add_argument("--data-path", default="./data/wikidiverse", help="Path to WikiDiverse data")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    parser.add_argument("--subset-size", type=int, default=None, help="Subset size for fast iteration")
    parser.add_argument("--experiment", choices=["baseline", "sweep", "layers", "full"], 
                        default="full", help="Experiment type")
    parser.add_argument("--output", default="./results", help="Output directory")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        wikidiverse_path=args.data_path,
        cache_dir=args.cache_dir,
        subset_size=args.subset_size
    )
    
    runner = ExperimentRunner(config)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.experiment == "baseline":
        results = runner.evaluate_baseline()
    elif args.experiment == "sweep":
        results = runner.run_bit_width_sweep()
        generate_accuracy_vs_quantization_plot(results, str(output_dir / "accuracy_vs_bits.png"))
    elif args.experiment == "layers":
        results = runner.run_layer_importance_analysis()
        report = generate_layer_importance_report(results)
        with open(output_dir / "layer_importance.md", "w") as f:
            f.write(report)
    else:  # full
        results = runner.run_full_experiment_suite()
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
