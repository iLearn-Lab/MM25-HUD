import datetime
import time
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F

from src.tools.files import json_dump, json_dump_append
from tqdm import tqdm
import json

class TestWebVidCoVR_FineGrained:
    def __init__(self, remove_self_similarity: bool = True, dataset: str = "covr"):
        self.remove_self_similarity = remove_self_similarity
        self.dataset = dataset

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for evaluation TestWebVidCoVR_FineGrained...")
        start_time = time.time()

        tar_img_feats = []
        tar_img_feats_mm = []
        tar_video_embeds = []
        tar_video_token_embeds = []
        query_feats = []
        query_feats_mm = []
        compose_feature_video_sentence_list = []
        compose_feature_token_list = []
        prob_compose_embedding = []
        prob_target_embedding = []
        captions = []
        pair_ids = []

        for batch in tqdm(data_loader):
            ref_img = batch["ref_img"]
            tar_img = batch["tar_img"]
            caption = batch["edit"]
            pair_id = batch["pair_id"]

            ref_description = batch["ref_description"]
            ref_webvid_caption = batch["ref_webvid_caption"]
            tag_description = batch["tag_description"]
            tag_webvid_caption = batch["tag_webvid_caption"]
            pair_ids.extend(pair_id.cpu().numpy().tolist())
            captions.extend(caption)

            device = ref_img.device
            query_feat, compose_feature_video_sentence, compose_feature_token, prob_compose, prob_compose_hol = model.compose_feature(ref_img, caption, ref_webvid_caption, fabric, device)
            query_feats.append(query_feat.cpu())
            prob_compose_embedding.append(prob_compose['embedding'].cpu())
            compose_feature_video_sentence_list.append(compose_feature_video_sentence.cpu())
            compose_feature_token_list.append(compose_feature_token.cpu())

            tar_feat, tar_img_feat_mm, tar_video_embed, tar_video_token_embed = model.target_fea(tar_img, tag_webvid_caption, fabric, device)
            # Encode the target image
            tar_img_feats.append(tar_feat.cpu())
            # prob_target_embedding.append(prob_target['embedding'].cpu())
            tar_video_embeds.append(tar_video_embed.cpu())
            tar_video_token_embeds.append(tar_video_token_embed.cpu())

        query_feats = torch.cat(query_feats, dim=0)
        tar_img_feats = torch.cat(tar_img_feats, dim=0)

        compose_feature_video_sentence_list = torch.cat(compose_feature_video_sentence_list, dim=0)
        compose_feature_token_list = torch.cat(compose_feature_token_list, dim=0)
        prob_compose_embedding = torch.cat(prob_compose_embedding, dim=0)
        # prob_target_embedding = torch.cat(prob_target_embedding, dim=0)
        tar_video_embeds = torch.cat(tar_video_embeds, dim=0)
        tar_video_token_embeds = torch.cat(tar_video_token_embeds, dim=0)

        query_feats = F.normalize(query_feats, dim=-1)
        # query_feats_mm = F.normalize(query_feats_mm, dim=-1)
        tar_img_feats = F.normalize(tar_img_feats, dim=-1)
        # tar_img_feats_mm = F.normalize(tar_img_feats_mm, dim=-1)
        compose_feature_video_sentence_list = F.normalize(compose_feature_video_sentence_list, dim=-1)
        compose_feature_token_list = F.normalize(compose_feature_token_list, dim=-1)
        tar_video_embeds = F.normalize(tar_video_embeds, dim=-1)
        tar_video_token_embeds = F.normalize(tar_video_token_embeds, dim=-1)
        prob_compose_embedding = F.normalize(prob_compose_embedding, dim=-1)
        # prob_target_embedding = F.normalize(prob_target_embedding, dim=-1)

        ref_img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
        tar_img_ids = [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]

        ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
        tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)
        if fabric.global_rank == 0:
            model = model.to("cpu")
            
            sim_q2t = model.fine_grained_sim(compose_feature_video_sentence_list, compose_feature_token_list, prob_compose_embedding, tar_video_embeds, tar_video_token_embeds, None).cpu().numpy()

            model = model.to(device)
            if self.remove_self_similarity:
                for i in range(len(ref_img_ids)):
                    for j in range(len(tar_img_ids)):
                        if ref_img_ids[i] == tar_img_ids[j]:
                            sim_q2t[i][j] = -10

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            recalls = eval_recall(sim_q2t)
            recalls["annotation"] = Path(data_loader.dataset.annotation_pth).name
            fabric.print(recalls)

            # Save results
            self_sim = "" if self.remove_self_similarity else "_ss"
            json_dump_append(recalls, f"recalls_{self.dataset}{self_sim}.json")

            print(
                f"Recalls saved in {Path.cwd()}/recalls_{self.dataset}{self_sim}.json"
            )

        fabric.barrier()

@torch.no_grad()
def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # type: ignore
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean3 = (tr1 + tr5 + tr10) / 3
    tr_mean4 = (tr1 + tr5 + tr10 + tr50) / 4

    eval_result = {
        "R1": round(tr1, 2),
        "R5": round(tr5, 2),
        "R10": round(tr10, 2),
        "R50": round(tr50, 2),
        "meanR3": round(tr_mean3, 2),
        "meanR4": round(tr_mean4, 2),
    }
    return eval_result
