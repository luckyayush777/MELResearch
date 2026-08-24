import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIMIC_ROOT = PROJECT_ROOT / "external" / "MIMIC_reproduction"
sys.path.insert(0, str(MIMIC_ROOT))

from codes.model.lightning_mimic import LightningForMIMIC
from codes.utils.dataset import DataModuleForMIMIC


def main():
    config_path = MIMIC_ROOT / "config" / "wikimel_clean_sanitized_visual_fp32_debug.yaml"
    args = OmegaConf.load(config_path)
    args.data.num_workers = 0

    pl.seed_everything(args.seed, workers=True)
    data_module = DataModuleForMIMIC(args)
    counts = {
        "train": len(data_module.train_data),
        "dev": len(data_module.val_data),
        "test": len(data_module.test_data),
        "entity": len(data_module.kb_entity),
        "qid": len(data_module.qid2id),
    }

    collator_checks = {}
    for name, collator, samples in (
        ("train", data_module.train_collator, data_module.train_data[:2]),
        ("eval", data_module.eval_collator, data_module.val_data[:2]),
        ("entity", data_module.entity_collator, data_module.kb_entity[:2]),
    ):
        before = copy.deepcopy(samples)
        first = collator(samples)
        second = collator(samples)
        collator_checks[name] = {
            "repeat_safe": samples == before,
            "first_batch_size": int(first["input_ids"].shape[0]),
            "second_batch_size": int(second["input_ids"].shape[0]),
            "has_missing_image_flag": "empty_img_flag" in first,
        }

    model = LightningForMIMIC(args).cuda()
    model.train()
    mention_batch = data_module.eval_collator(data_module.train_data[:2])
    mention_batch.pop("answer")
    mention_batch["empty_img_flag"] = torch.tensor([False, True])
    mention_batch = {key: value.cuda() for key, value in mention_batch.items()}
    entity_batch = data_module.entity_collator(data_module.kb_entity[:2])
    entity_batch["empty_img_flag"] = torch.tensor([False, True])
    entity_batch = {key: value.cuda() for key, value in entity_batch.items()}

    mention_outputs = model.encoder(**mention_batch)
    entity_outputs = model.encoder(**entity_batch)
    encoder_mask_zero = bool(
        torch.count_nonzero(mention_outputs[1][1]).item() == 0
        and torch.count_nonzero(mention_outputs[3][1]).item() == 0
        and torch.count_nonzero(entity_outputs[1][1]).item() == 0
        and torch.count_nonzero(entity_outputs[3][1]).item() == 0
    )
    scores, components = model.matcher(
        entity_outputs[0], entity_outputs[2], mention_outputs[0], mention_outputs[2],
        entity_outputs[1], entity_outputs[3], mention_outputs[1], mention_outputs[3],
        entity_batch["empty_img_flag"], mention_batch["empty_img_flag"],
    )
    text_scores, image_scores, cross_modal_scores = components
    invalid_pairs = torch.tensor([[False, True], [True, True]], device=scores.device)
    matcher_mask_zero = bool(
        torch.count_nonzero(image_scores[invalid_pairs]).item() == 0
        and torch.count_nonzero(cross_modal_scores[invalid_pairs]).item() == 0
        and torch.allclose(scores[invalid_pairs], text_scores[invalid_pairs])
    )
    del mention_outputs, entity_outputs, scores, components

    torch.cuda.reset_peak_memory_stats()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(model, datamodule=data_module)

    result = {
        "canonical_multiline_qid_load": counts["qid"] == counts["entity"],
        "counts": counts,
        "collators": collator_checks,
        "encoder_missing_image_zeroed": encoder_mask_zero,
        "matcher_missing_visual_terms_zeroed_and_text_only": matcher_mask_zero,
        "training_batch_size": int(args.data.batch_size),
        "one_batch_fp32_cuda_fit": trainer.state.finished,
        "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
        "run_test_after_fit": bool(args.run_test_after_fit),
    }
    print("MIMIC_FIX_VERIFICATION=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
