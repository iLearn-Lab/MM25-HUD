"""
Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from src.model.blip2.blip2 import Blip2Base, disabled_train
import numpy as np
from pathlib import Path




class HUD(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        loss: Any,
        vit_model="clip_L",
        image_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        train_vit=False,
        vit="large",
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        temperature=1,
        si_ti_weight=1,
        si_tc_weight=0,
        num_frames=1,
        n_video_samples=7,

    ):
        super().__init__()
        print("Initialized HUD")
        print(num_frames)
        print(n_video_samples)
        self.loss = loss

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.train_vit = train_vit
        if not train_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = temperature

        self.max_txt_len = max_txt_len
        self.ifFrames = False # if input has multiple frames, default to False

        assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"
        self.si_ti_weight = si_ti_weight
        self.si_tc_weight = si_tc_weight

        # @ MLP Composer
        composer_dim = embed_dim
        self.composer_video = nn.Sequential(nn.Conv1d(composer_dim * 2, composer_dim, kernel_size=1, stride=1, bias=True),
                                    nn.LeakyReLU(),
                                    nn.Conv1d(composer_dim, composer_dim * 2, kernel_size=1, stride=1, bias=True),
                                    nn.Sigmoid(),
                                )
        
        self.composer_uncertainty = nn.Sequential(nn.Conv1d(composer_dim * 2, composer_dim, kernel_size=1, stride=1, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Conv1d(composer_dim, composer_dim * 2, kernel_size=1, stride=1, bias=True),
                                            nn.Sigmoid(),
                                        )
        self.n_video_samples = n_video_samples
        self.num_frames = num_frames
        self.video_logit_weight = nn.parameter.Parameter(torch.eye(1 + self.n_video_samples), requires_grad=True)

        self.compose_token_mat_weight = nn.parameter.Parameter(torch.eye(num_query_token * self.num_frames + num_query_token), requires_grad=True)

        Uncertainty_dim = embed_dim#self.Qformer.config.hidden_size
        self.probabilistic_cross = Probabilistic_Cross(1, Uncertainty_dim, Uncertainty_dim, Uncertainty_dim // 2)
        self.uncertain_net_cross = UncertaintyModule_Cross(Uncertainty_dim, Uncertainty_dim, Uncertainty_dim // 2)
        self.probabilistic_cross_Token = Probabilistic_Cross(1, self.Qformer.config.hidden_size, self.Qformer.config.hidden_size, self.Qformer.config.hidden_size // 2)
        self.uncertain_net_cross_Token = UncertaintyModule_Cross(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size, self.Qformer.config.hidden_size // 2)


        self.local_weight = nn.Parameter(torch.tensor([1.0 for _ in range(num_query_token * self.num_frames + num_query_token)]))
        self.local_weight_hol = nn.Parameter(torch.tensor([1.0 for _ in range(1 + self.n_video_samples)]))
        self.t = 0.1


    def holistic_compose(self, img_token, text_token, composer, device='cuda'):
        compose_embed_ = torch.cat([img_token, text_token], dim=-1).transpose(1, 2)
        batch = img_token.shape[0]
        chanel = img_token.shape[1] * 2
        gamma, beta = composer(compose_embed_).transpose(1, 2).reshape(batch, chanel, -1).chunk(2, dim=1)
        x = gamma * img_token + (beta) * text_token
        return x

    def target_fea(self, tar_img, description, fabric=None, device='cuda'):
        if tar_img.dim() == 5:
            self.ifFrames = True

        bs = tar_img.shape[0]
        nf = 1

        if self.ifFrames:
            bs, nf, c, h, w = tar_img.shape
            tar_img = tar_img.view(bs * nf, c, h, w)

        if self.train_vit:
            tar_img_embs = self.ln_vision(self.visual_encoder(tar_img))
        else:
            with torch.no_grad():
                tar_img_embs = self.ln_vision(self.visual_encoder(tar_img))
        tar_img_embs = tar_img_embs.float()

        query_tokens = self.query_tokens.expand(
            bs, -1, -1
        )

        if self.ifFrames:
            tar_img_embs = tar_img_embs.view(bs, -1, tar_img_embs.shape[-1])
            query_tokens = query_tokens.repeat(1, nf, 1)


        tar_img_atts = torch.ones(tar_img_embs.size()[:-1], dtype=torch.long).to(device)
        tar_img_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=tar_img_embs,
            encoder_attention_mask=tar_img_atts,
            return_dict=True,
        )
        tar_embs_ = tar_img_output.last_hidden_state[:, : query_tokens.size(1), :]
        tar_embs_proj =  F.normalize(self.vision_proj(tar_embs_), dim=-1)

        video_embeds = tar_embs_proj.mean(dim=1).unsqueeze(1)
        video_token_embeds = tar_embs_proj
        tar_embs = torch.cat([video_embeds, video_token_embeds], dim=1)


        target_si_feat_ = F.normalize(tar_embs, dim=-1)

        if description[0] != "":
            des_out = self.textual_feature(description)
        else:
            des_out = None
        tar_video_embeds = video_embeds
        tar_video_token_embeds = video_token_embeds
        tar_video_embeds = torch.cat([tar_video_embeds, tar_video_embeds.repeat(1, self.n_video_samples, 1)], dim=1)

        tar_video_token_embeds = torch.cat([tar_video_token_embeds, tar_video_token_embeds], dim=1)
        return target_si_feat_.mean(dim=1), des_out, F.normalize(tar_video_embeds, dim=-1),  F.normalize(tar_video_token_embeds, dim=-1)
    
    def compose_feature(self, ref_img, caption, description, fabric=None, device='cuda'):
        if ref_img.dim() == 5:
            self.ifFrames = True
        bs = ref_img.shape[0]
        nf = 1

        text_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_embeds = text_output.last_hidden_state[:, : 32, :]

        text_embed_proj = F.normalize(self.text_proj(text_embeds), dim=-1)

        if self.ifFrames:
            bs, nf, c, h, w = ref_img.shape
            ref_img = ref_img.view(bs * nf, c, h, w)

        if self.train_vit:
            ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))
        else:
            with torch.no_grad():
                ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))

        query_tokens = self.query_tokens.expand(bs, -1, -1)

        if self.ifFrames:
            ref_img_embs = ref_img_embs.view(bs, -1, ref_img_embs.shape[-1])
            query_tokens = query_tokens.repeat(1, nf, 1)

        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)
        ref_img_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        ref_embs = ref_img_output.last_hidden_state[:, : query_tokens.size(1), :]
        ref_emb_proj = F.normalize(self.vision_proj(ref_embs), dim=-1)

        video_embeds = ref_emb_proj.mean(dim=1).unsqueeze(1)
        sentence_embeds = text_embed_proj.mean(dim=1).unsqueeze(1)


        compose_hol_feature = self.holistic_compose(video_embeds, sentence_embeds, self.composer_video, device)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        compose_atomistic_feature_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        compose_atomistic_feature = compose_atomistic_feature_output.last_hidden_state[:, : query_tokens.size(1), :]
        compose_atomistic_feature = F.normalize(self.text_proj(compose_atomistic_feature), dim=-1)
        
        compose_feature = torch.cat([compose_hol_feature, compose_atomistic_feature], dim=1)
        
        query_si_feat_ = F.normalize(compose_feature, dim=-1)

        compose_hol_feature_ = compose_hol_feature 

        compose_atomistic_feature_ = compose_atomistic_feature


        # Holistic Uncertainty Reasoning
        prob_modification_holistic_hol = self.probabilistic_cross_modal(sentence_embeds.squeeze(1), video_embeds, self.probabilistic_cross, self.uncertain_net_cross)

        prob_compose_hol_embedding = self.holistic_compose(video_embeds.repeat(1, prob_modification_holistic_hol["embedding"].shape[1], 1), prob_modification_holistic_hol["embedding"], self.composer_uncertainty, device)

        compose_hol_feature_ = torch.cat([compose_hol_feature_, prob_compose_hol_embedding], dim=1)
        prob_compose_hol = {"embedding": prob_modification_holistic_hol["embedding"], "logsigma": prob_modification_holistic_hol["logsigma"], "compose": prob_compose_hol_embedding}

        prob_ref_detail = self.probabilistic_cross_modal(ref_embs, text_embeds, self.probabilistic_cross_Token, self.uncertain_net_cross_Token)

        prob_compose_embedding = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=prob_ref_detail["embedding"],
                attention_mask=attention_mask,
                return_dict=True,
            )
        prob_compose_embedding = prob_compose_embedding.last_hidden_state[:, : query_tokens.size(1), :]
        prob_compose_embedding = F.normalize(self.text_proj(prob_compose_embedding), dim=-1)

        prob_compose = {"embedding": prob_ref_detail["embedding"], "logsigma": prob_ref_detail["logsigma"], "compose_U": prob_compose_embedding, "compose": compose_atomistic_feature_, "ref_embs": ref_embs, "text_embeds": text_embeds}
        compose_atomistic_feature_  = torch.cat([compose_atomistic_feature_, prob_compose_embedding], dim=1)
        return query_si_feat_, F.normalize(compose_hol_feature_, dim=-1), F.normalize(compose_atomistic_feature_, dim=-1), prob_compose, prob_compose_hol

    def textual_feature(self, caption, device='cuda'):

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        text_output = self.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        text_embeds = text_output.last_hidden_state
        text_features = self.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)
        return text_features#.mean(dim=1)


    def visual_feature(self, tar_img, device='cuda'):
        if self.train_vit:
            tar_img_embs = self.ln_vision(self.visual_encoder(tar_img))
        else:
            with torch.no_grad():
                tar_img_embs = self.ln_vision(self.visual_encoder(tar_img))
        tar_img_embs = tar_img_embs.float()
        query_tokens = self.query_tokens.expand(
            tar_img_embs.shape[0], -1, -1
        )
        tar_img_atts = torch.ones(tar_img_embs.size()[:-1], dtype=torch.long).to(device)
        tar_img_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=tar_img_embs,
            encoder_attention_mask=tar_img_atts,
            return_dict=True,
        )
        vl_embs = tar_img_output.last_hidden_state[:, : query_tokens.size(1), :]
        target_si_feat = F.normalize(self.vision_proj(vl_embs), dim=-1)

        return target_si_feat.mean(dim=1)


    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)
        samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(
            mu.unsqueeze(1))
        samples = F.normalize(samples, p=2, dim=-1) 
        return samples
    

    def sample_gaussian_tensors_3d(self, mu, logsigma):
        eps = torch.randn(mu.size(0), mu.size(1), mu.size(2), dtype=mu.dtype, device=mu.device)

        samples = eps.mul(torch.exp(logsigma)).add_(
            mu)
        return samples



    def probabilistic_cross_modal(self, modal_in, inter_model_in, probabilistic_cross, uncertain_net_cross):
        output = {}

        out, attn, residual = probabilistic_cross(modal_in, inter_model_in)    
        output['attention'] = attn
        output['residual'] = residual  

        uncertain_out = uncertain_net_cross(modal_in, inter_model_in) 
        logsigma = uncertain_out['logsigma'] 
        output['logsigma'] = logsigma  
        output['uncertainty_attention'] = uncertain_out['attention']

        out = out 
        output['mean'] = out   
        if len(out.shape) == 2:
            output['embedding'] = F.normalize(self.sample_gaussian_tensors(out, logsigma, self.n_video_samples), dim=-1) 
        else:
            output['embedding'] = F.normalize(self.sample_gaussian_tensors_3d(out, logsigma), dim=-1)  

        return output


    def fine_grained_sim(self, compose_hol_feature, compose_atomistic_feature, prob_compose_embedding, tar_video_embeds, tar_video_token_embeds, prob_target_embedding):

        compose_hol_feature = compose_hol_feature.unsqueeze(1)
        tar_video_embeds = tar_video_embeds.permute(0, 2, 1)

        similarity_results = []
        for i in range(compose_hol_feature.size(0)):
            single_compose_hol_feature = compose_hol_feature[i].unsqueeze(0)
            similarity = torch.matmul(single_compose_hol_feature, tar_video_embeds)
            
            # 取最大值
            max_similarity = similarity.max(-1)[0] 
            max_similarity = max_similarity.cuda()
            holistic_logits_ = torch.sum(max_similarity \
                * torch.softmax(torch.matmul(torch.softmax(max_similarity / 1e-2, dim=-1), self.video_logit_weight.cuda()) / 1e-2, dim=-1), dim=-1) # .t()
            
            # 将结果添加到列表中
            similarity_results.append(holistic_logits_.cpu())

        holistic_logits = torch.cat(similarity_results, dim=0)
        print("fine_grained_sim tmp_logits: ", holistic_logits.shape)
        compose_atomistic_feature = compose_atomistic_feature.unsqueeze(1).cuda()
        tar_video_token_embeds = tar_video_token_embeds.permute(0, 2, 1).cuda()
        print("Token sims")
        similarity_results = []
        for i in range(compose_atomistic_feature.size(0)):
            single_compose_atomistic_feature = compose_atomistic_feature[i].unsqueeze(0) 
            similarity = torch.matmul(single_compose_atomistic_feature, tar_video_token_embeds) 
            max_similarity = similarity.max(-1)[0] 
            max_similarity = max_similarity.cuda()
            atomistic_logits_ = torch.sum(max_similarity \
            * torch.softmax(torch.matmul(torch.softmax(max_similarity / 1e-2, dim=-1), self.compose_token_mat_weight.cuda()) / 1e-2, dim=-1), dim=-1)
        
            similarity_results.append(atomistic_logits_.cpu())
        print("Token sims done")
        atomistic_tmp_logits = torch.cat(similarity_results, dim=0)
        logits = (holistic_logits + atomistic_tmp_logits) / 2

        return logits




    def forward(self, batch, fabric):
        ref_img = batch["ref_img"]
        tar_img = batch["tar_img"]
        caption = batch["edit"]

        ref_webvid_caption = batch["ref_webvid_caption"]
        tag_webvid_caption = batch["tag_webvid_caption"]
        ref_img.half()
        tar_img.half()

        if ref_img.dim() == 5:
            self.ifFrames = True


        device = ref_img.device
        _, compose_hol_feature, compose_atomistic_feature, prob_ref_detail, prob_modification_holistic = self.compose_feature(ref_img, caption, ref_webvid_caption, fabric, device)
        _, _, tar_video_embeds, tar_video_token_embeds = self.target_fea(tar_img, tag_webvid_caption, fabric, device)
        

        ### Holistic-to-Atomistic Alignment
        bs = compose_hol_feature.size(0)
        tmp_logits = torch.matmul(compose_hol_feature.unsqueeze(1), tar_video_embeds.permute(0, 2, 1)).max(-1)[0] #B*1*F_c*D, B*D*F_t-->B*B*F_c*F_t

        holistic_logits = torch.sum(tmp_logits \
            * torch.softmax(torch.matmul(torch.softmax(tmp_logits / 1e-2, dim=-1), self.video_logit_weight) / 1e-2, dim=-1), dim=-1)#.t()

        atomistic_tmp_logits = torch.matmul(compose_atomistic_feature.unsqueeze(1), tar_video_token_embeds.permute(0, 2, 1)).max(-1)[0]
        atomistic_logits = torch.sum(atomistic_tmp_logits \
            * torch.softmax(torch.matmul(torch.softmax(atomistic_tmp_logits / 1e-2, dim=-1), self.compose_token_mat_weight) / 1e-2, dim=-1), dim=-1)#.t()

        logits = (holistic_logits + atomistic_logits) / 2

        bs = holistic_logits.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(
            holistic_logits.device
        )
        sim_i2t = logits / self.temp
        loss_itc = F.cross_entropy(sim_i2t, targets)


        loss = {}
        if self.si_ti_weight > 0:
            si_ti_loss = loss_itc
            loss["rank"] = si_ti_loss * self.si_ti_weight
            compose_token = (F.normalize(compose_atomistic_feature, p=2, dim=-1) * self.local_weight.unsqueeze(0).unsqueeze(-1)).flatten(1)
            target_token = (F.normalize(tar_video_token_embeds, p=2, dim=-1) * self.local_weight.unsqueeze(0).unsqueeze(-1)).flatten(1)

            compose_hol_token = (F.normalize(compose_hol_feature, p=2, dim=-1) * self.local_weight_hol.unsqueeze(0).unsqueeze(-1)).flatten(1)
            target_hol_token = (F.normalize(tar_video_embeds, p=2, dim=-1) * self.local_weight_hol.unsqueeze(0).unsqueeze(-1)).flatten(1)
            loss["kl"] = 0.5 * (self.kl_div(compose_hol_token, target_hol_token, target_token, target_token)\
            + self.kl_div(compose_token, target_token, target_hol_token, target_hol_token)) 

        return loss

    def kl_div(self, x1, y1, x2, y2, t=0.1):
        x1 = F.normalize(x1, p=2, dim=-1)
        y1 = F.normalize(y1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        y2 = F.normalize(y2, p=2, dim=-1)

        x1_y1 = torch.mm(x1, y1.T) / t
        x2_y2 = torch.mm(x2, y2.T) / t

        log_soft_x1 = F.log_softmax(x1_y1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2_y2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')

        return kl



class UncertaintyModule_Cross(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadCrossAttention(1, d_in, d_out, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        if len(out.shape) == 2:
            residual, attn = self.attention(out.unsqueeze(1), x, x)
            fc_out = self.fc2(out)
            out = self.fc(residual.squeeze(1)) + fc_out
        else:
            residual, attn = self.attention(out, x, x, mask=pad_mask)
            fc_out = self.fc2(out)
            out = self.fc(residual) + fc_out


        return {
            'logsigma': out,
            'attention': attn,
        }


class MultiHeadCrossAttention(nn.Module):
    """Cross-Attention module."""

    def __init__(self, n_head, d_query, d_key, d_hidden):
        super(MultiHeadCrossAttention, self).__init__()
        
        self.n_head = n_head
        self.w_q = nn.Linear(d_query, d_hidden, bias=False)
        self.w_k = nn.Linear(d_key, d_hidden, bias=False)
        self.w_v = nn.Linear(d_key, d_hidden, bias=False)
        self.w_o = nn.Linear(d_hidden, d_query, bias=False)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
    
    def forward(self, query, key, value, mask=None):
        # query: (batch_size, query_len, d_query)
        # key, value: (batch_size, key_len, d_key)
        q_proj = self.w_q(query)  # (batch_size, query_len, d_hidden)
        k_proj = self.w_k(key)    # (batch_size, key_len, d_hidden)
        v_proj = self.w_v(value)  # (batch_size, key_len, d_hidden)
        
        attn_scores = torch.bmm(q_proj, k_proj.transpose(1, 2))  # (batch_size, query_len, key_len)
        
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, -np.inf)
        
        attn_weights = self.softmax(attn_scores)  # (batch_size, query_len, key_len)
        
        output = torch.bmm(attn_weights, v_proj)  # (batch_size, query_len, d_hidden)
        output = self.w_o(output)  # (batch_size, query_len, d_query)
        
        return output, attn_weights


class Probabilistic_Cross(nn.Module):
    """Polysemous Instance Embedding (PIE) module"""

    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.0):
        super(Probabilistic_Cross, self).__init__()

        self.num_embeds = n_embeds
        self.attention = MultiHeadCrossAttention(n_embeds, d_in, d_out, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x, pad_mask=None):
        if len(out.shape) == 2:
            residual, attn = self.attention(out.unsqueeze(1), x, x)
            residual = self.dropout(self.sigmoid(self.fc(residual.squeeze(1))))
        else:
            residual, attn = self.attention(out, x, x)
            residual = self.dropout(self.sigmoid(self.fc(residual)))
        # if self.num_embeds > 1:
        #     out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
        out = self.layer_norm(out + residual)
        return out, attn, residual



def hud(model, ckpt_path, **kwargs):
    if ckpt_path:
        model.load_from_pretrained(url_or_filename=ckpt_path)
    return model
