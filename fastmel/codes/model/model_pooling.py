import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel

class MLPWeightedPooling(nn.Module):
    def __init__(self, d_model, hidden=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, mask=None, return_weights=False):
        # x: [B, L, D]
        scores = self.mlp(x).squeeze(-1)  # [B, L]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)

        if return_weights:
            return pooled, x, weights
        else:
            return pooled
        

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.clip_model = AutoModel.from_pretrained(args.pretrained_model)
        self.text_encoder = self.clip_model.text_model
        self.vision_encoder = self.clip_model.vision_model

        # Projection to shared embed space
        text_dim = self.text_encoder.config.hidden_size
        vision_dim = self.vision_encoder.config.hidden_size

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, args.model.embed_dim)
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, args.model.embed_dim)
        )

        self.pooling_type = args.model.pooling_type

        if self.pooling_type == 'mlp' or self.pooling_type == 'cls-mlp':
            self.mlp_pooling = MLPWeightedPooling(d_model=args.model.embed_dim,hidden=args.model.embed_dim//2)

        if self.pooling_type == 'mean-attention':
            self.attention_mean_pooling = AttentionMeanPooling(d_model=args.model.embed_dim)
        if self.pooling_type == 'cross-attention':
            self.cross_attention_pooling = CrossAttentionPooling(d_model=args.model.embed_dim)

        if self.pooling_type == 'cross-attention-mlp':
            self.cross_attention_pooling = CrossAttentionPoolingMLP(d_model=args.model.embed_dim)

        if self.pooling_type == 'learned-attention':
            self.learned_attention_pooling = LearnedAttentionPooling(dim=args.model.embed_dim)

    def forward(self, input_ids, pixel_values, attention_mask=None, return_weights=False, empty_img_flag=None):
        B = pixel_values.size(0)

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        vision_out = self.vision_encoder(pixel_values=pixel_values, return_dict=True)

        text_hidden = self.text_proj(text_out.last_hidden_state)      # [B, T, D]
        vision_hidden = self.vision_proj(vision_out.last_hidden_state)  # [B, P, D]

        B, P, D = vision_hidden.shape
        T = text_hidden.shape[1]
        
        if self.args.model.use_vision : 
            image_mask = empty_img_flag.unsqueeze(1).repeat(1, P).to("cuda")  # [B, P], bool
            mask = torch.cat([
               ~image_mask,  
               attention_mask.type(torch.bool)
            ], dim=1)  # [B, P+T]
        else :
            mask= torch.cat([torch.ones_like(empty_img_flag).unsqueeze(1).repeat(1, P), attention_mask], dim=1)  # [B, P+T]

        concat_features = torch.cat([vision_hidden, text_hidden], dim=1)  # [B, P+T, D]

   

    

        if self.pooling_type == 'mean':
            masked_features = concat_features * mask.unsqueeze(-1)   
        
            sum_features = masked_features.sum(dim=1)  # [B, D]
            lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B, 1]
            return sum_features / lengths

        
        elif self.pooling_type == 'max':
            mask = mask.unsqueeze(-1)  # [B, T, 1]

            masked_features = concat_features.masked_fill(
                mask == 0, float('-inf')
            )

            return masked_features.max(dim=1).values  # [B, D]

        

        if self.pooling_type == 'mlp':
            return self.mlp_pooling(concat_features,
                                            mask=mask,
                                            return_weights=return_weights)


        elif self.pooling_type == 'cls':
            cls_tokens = torch.stack([text_hidden[:,0,:], vision_hidden[:,0,:]], dim=1)  # [B, 2, D]
            return torch.mean(cls_tokens, dim=1)  # [B, D]
        elif self.pooling_type == 'cls-mlp':
            cls_tokens = torch.stack([text_hidden[:,0,:], vision_hidden[:,0,:]], dim=1)  # [B, 2, D]
            return self.mlp_pooling(cls_tokens, mask=None,
                                            return_weights=return_weights)  # [B, D]
        elif self.pooling_type == 'min':
            return torch.min(concat_features, dim=1).values  # [B, D]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
