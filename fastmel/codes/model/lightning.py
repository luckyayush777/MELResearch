import json
import os
import time
import math
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from codes.model.model_pooling import Encoder

# model_late_int.py is not present in this repository — it is a leftover from
# the authors' larger private codebase. It is only needed for the late-interaction
# variant (model.pooling_type left empty), which none of the shipped configs use,
# so import it lazily instead of failing at module load.
try:
    from codes.model.model_late_int import LateEncoder
except ImportError:
    LateEncoder = None

def l2norm(x): return torch.nn.functional.normalize(x, dim=-1)

def cosine_similarity(x1, x2):
    x1 = l2norm(x1)
    x2 = l2norm(x2)
    
    return torch.matmul(x1, x2.T)



class Lightning(pl.LightningModule):
    def __init__(self, args):
        super(Lightning, self).__init__()
        self.args = args

        if args.model.pooling_type:
            self.encoder = Encoder(args)
        elif LateEncoder is None:
            raise ImportError(
                "model.pooling_type is empty, which selects the late-interaction "
                "encoder, but codes/model/model_late_int.py is missing from this "
                "repository. Set model.pooling_type (the shipped configs use 'mlp')."
            )
        else:
            self.encoder = LateEncoder(args)

        self.loss_fct = torch.nn.CrossEntropyLoss()
        # Upstream shipped this as an incomplete ternary with no `else`, which is a
        # SyntaxError — the module could not be imported at all. cosine_similarity is
        # the only scoring function defined in the repo, so anything else is rejected.
        if args.score_function == "cosine":
            self.score_function = cosine_similarity
        else:
            raise ValueError(
                "unsupported score_function %r; only 'cosine' is implemented"
                % (args.score_function,)
            )
        print("score_function:", self.score_function)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))  # init with 0.07
        self.save_hyperparameters(args)

    def training_step(self, batch, batch_idx):
        ent_batch = {}
        cands_batch = {}
        mention_batch = {}
        for k, v in batch.items():
            if k.startswith('ent_'):
                ent_batch[k.replace('ent_', '')] = v
            elif k.startswith('cands_'):
                cands_batch[k.replace('cands_', '')] = v
            else:
                mention_batch[k] = v
        entity_empty_image_flag = ent_batch.pop('empty_img_flag')  
        mention_empty_image_flag = mention_batch.pop('empty_img_flag')
        cands_empty_image_flag = cands_batch.pop('empty_img_flag') if cands_batch else None

        ent_embeddings = self.encoder(**ent_batch, empty_img_flag=entity_empty_image_flag)
        mention_embeddings = self.encoder(**mention_batch, empty_img_flag=mention_empty_image_flag)
        if cands_batch :
            candidates_embeddings = self.encoder(**cands_batch, empty_img_flag=cands_empty_image_flag)

        if cands_batch :
            entities_and_cands = torch.cat([ent_embeddings, candidates_embeddings], dim=0)

        else :  
            entities_and_cands = ent_embeddings

        logit_scale = self.logit_scale.exp().clamp(max=100)

        
        labels_mentions = torch.arange(mention_embeddings.size(0), device=mention_embeddings.device)
        
        logits_cross = logit_scale * self.score_function(mention_embeddings, entities_and_cands)
        loss = self.loss_fct(logits_cross, labels_mentions)
        self.log('Train/loss_total', loss.detach().cpu().item(), on_epoch=True, prog_bar=True)

        return loss


    def _step(self, batch):
        answer = batch.pop('answer')
        _ = batch.pop('candidates', None)
        empty_img_flag = batch.pop('empty_img_flag', None)
        batch_size = len(answer)
        mentions_embeds = self.encoder(**batch, empty_img_flag=empty_img_flag)

        # chunk scoring
        scores = []
        chunk_size = self.args.data.eval_chunk_size
        num_chunks = math.ceil(self.args.data.num_entity / chunk_size)

        for idx in range(num_chunks):
            start_pos = idx * chunk_size
            end_pos = (idx + 1) * chunk_size
            
           
            chunk_entity_embeds = self.entity_embeds[start_pos:end_pos].to(mentions_embeds.device)

            chunk_score = self.score_function(mentions_embeds, chunk_entity_embeds)
            scores.append(chunk_score)

        scores = torch.cat(scores, dim=-1)

        # compute rank
        rank = torch.argsort(torch.argsort(scores, dim=-1, descending=True),
                            dim=-1, descending=False) + 1

        tgt_rank = rank[torch.arange(batch_size), answer].detach().cpu()
        return dict(rank=tgt_rank, all_rank=rank.detach().cpu().numpy())


    def _update_entity_embeds(self):
        entity_dataloader = self.trainer.datamodule.entity_dataloader()
        outputs = []

        with torch.no_grad():
            for batch in tqdm(entity_dataloader, desc="UpdateEmbed", total=len(entity_dataloader)):
                batch = pl.utilities.move_data_to_device(batch, self.device)
                
                entity_embeds = self.encoder(**batch)
                if self.args.model.reduction == 'all_maxsim' :
                    outputs.append((entity_embeds[0].cpu(), entity_embeds[1].cpu()))
                else :
                    outputs.append(entity_embeds.cpu())
        if self.args.model.reduction == 'all_maxsim' :
            self.entity_embeds = (torch.cat([o[0] for o in outputs], dim=0),
                                  torch.cat([o[1] for o in outputs], dim=0))
        else :
            self.entity_embeds = torch.cat(outputs, dim=0)


    def _compute_epoch_metrics(self, outputs, prefix):
        self.entity_embeds = None

        ranks = np.concatenate([_['rank'] for _ in outputs])

        self.log(f"{prefix}/hits20", (ranks <= 20).mean())
        self.log(f"{prefix}/hits10", (ranks <= 10).mean())
        self.log(f"{prefix}/hits5",  (ranks <= 5).mean())
        self.log(f"{prefix}/hits3",  (ranks <= 3).mean())
        self.log(f"{prefix}/hits1",  (ranks <= 1).mean())
        self.log(f"{prefix}/mr", ranks.mean())
        self.log(f"{prefix}/mrr", (1. / ranks).mean())


    def validation_step(self, batch, batch_idx):
        return self._step(batch)

    def test_step(self, batch, batch_idx):
        return self._step(batch)

    def on_validation_start(self):
        self._update_entity_embeds()

    def on_test_start(self):
        self._update_entity_embeds()

    def validation_epoch_end(self, outputs):
        self._compute_epoch_metrics(outputs, prefix="Val")

    def test_epoch_end(self, outputs):
        self._compute_epoch_metrics(outputs, prefix="Test")


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {
                'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.0001
            },
            {
                'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_params,
            lr=self.args.lr,
            betas=(0.9, 0.999),
            eps=1e-4
        )
        return [optimizer]
