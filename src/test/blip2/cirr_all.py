import datetime
import time
from collections import OrderedDict
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning

import numpy as np
import torch
import torch.nn.functional as F
import logging
from src.tools.utils import concat_all_gather
from tqdm import tqdm
from src.tools.files import json_dump


class CIRR_ALL:
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def __call__(model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for validation Original Test...")
        start_time = time.time()
        test_queries, test_targets = data_loader.dataset.val_queries, data_loader.dataset.val_targets

        vl_feats = []
        pair_ids = []

        tar_img_feats = []
        tar_video_embeds = []
        tar_video_token_embeds = []
        query_feats = []
        compose_feature_video_sentence_list = []
        compose_feature_token_list = []
        prob_compose_embedding = []
        prob_target_embedding = []
        captions = []
        device = "cuda"
        with torch.no_grad():   
            all_queries = []
            all_imgs = []
            if test_queries:
                # compute test query features
                imgs = []
                mods = []
                for t in tqdm(test_queries):
                    imgs += [t['source_img_data']]
                    mods += [t['mod']['str']]
                    if len(imgs) >= data_loader.batch_size or t is test_queries[-1]:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float().cuda()
                        # mods = [txt_processors["eval"](caption) for caption in mods]
                        query_feat, compose_feature_video_sentence, compose_feature_token, prob_compose, prob_compose_hol = model.compose_feature(imgs, mods, [""], fabric, device)
                        prob_compose_embedding.append(prob_compose['embedding'].cpu())
                        compose_feature_video_sentence_list.append(compose_feature_video_sentence.cpu())
                        compose_feature_token_list.append(compose_feature_token.cpu())
                        imgs = []
                        mods = []

                # compute all image features
                imgs = []
                for t in tqdm(test_targets):
                    imgs += [t['target_img_data']]
                    if len(imgs) >= data_loader.batch_size or t is test_targets[-1]:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float().cuda()
                        tar_feat, tar_img_feat_mm, tar_video_embed, tar_video_token_embed = model.target_fea(imgs, [""], fabric, device)
                        # prob_target_embedding.append(prob_target['embedding'].cpu())
                        tar_video_embeds.append(tar_video_embed.cpu())
                        tar_video_token_embeds.append(tar_video_token_embed.cpu())
                        imgs = []

                compose_feature_video_sentence_list = torch.cat(compose_feature_video_sentence_list, dim=0)
                compose_feature_token_list = torch.cat(compose_feature_token_list, dim=0)

                prob_compose_embedding = torch.cat(prob_compose_embedding, dim=0)
                # prob_target_embedding = torch.cat(prob_target_embedding, dim=0)
                tar_video_embeds = torch.cat(tar_video_embeds, dim=0)
                tar_video_token_embeds = torch.cat(tar_video_token_embeds, dim=0)

                model = model.to("cpu")
                # sim_q2t = model.fine_grained_sim(compose_feature_video_sentence_list, compose_feature_frame_sentence_list, compose_feature_token_list, tar_frame_embeds, tar_video_embeds, tar_video_token_embeds).cpu().numpy()
                sims = model.fine_grained_sim(compose_feature_video_sentence_list, compose_feature_token_list, prob_compose_embedding, tar_video_embeds, tar_video_token_embeds, None).cpu().numpy()

                model = model.to(device)
                print("model moved to cpu")

                test_targets_id = []
                for i in test_targets:
                    test_targets_id.append(i['target_img_id'])
                for i, t in enumerate(test_queries):
                    sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


                nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])] # (m,n)

                # all set recalls
                cirr_out = []
                for k in [1, 5, 10, 50]:
                    r = 0.0
                    for i, nns in enumerate(nn_result):
                        if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                            r += 1
                    r = 100 * r / len(nn_result)
                    cirr_out += [('{}_r{}'.format("cirr",k), r)]

                # subset recalls
                for k in [1, 2, 3]:
                    r = 0.0
                    for i, nns in enumerate(nn_result):

                        subset = np.array([test_targets_id.index(idx) for idx in test_queries[i]['subset_id']]) # (6)
                        subset_mask = (nns[..., None] == subset[None, ...]).sum(-1).astype(bool) # (n,1)==(1,6) => (n,6) => (n)
                        subset_label = nns[subset_mask] # (6)
                        if test_targets_id.index(test_queries[i]['target_img_id']) in subset_label[:k]:
                            r += 1
                    r = 100 * r / len(nn_result)
                    cirr_out += [('{}_subset_r{}'.format("cirr", k), r)]

                logging.info(cirr_out)


        ###Test Json Generate
        test_queries = data_loader.dataset.test_queries

        vl_feats = []
        pair_ids = []

        tar_img_feats = []
        tar_video_embeds = []
        tar_video_token_embeds = []
        query_feats = []
        compose_feature_video_sentence_list = []
        compose_feature_token_list = []
        prob_compose_embedding = []
        prob_target_embedding = []
        captions = []
        device = "cuda"
        with torch.no_grad():   
            all_queries = []
            imgs = []
            mods = []
            pairid = []
            subset = []
            reference_name = []
            for i, data in enumerate(tqdm(test_queries)):
                imgs += [data['reference_data']]
                mods += [data['mod']]
                pairid += [data['pairid']]
                reference_name += [data['reference_name']]
                subset.append(list(data['subset']))
                if len(imgs) >= data_loader.batch_size or i == len(test_queries) - 1:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    query_feat, compose_feature_video_sentence, compose_feature_token, prob_compose, prob_compose_hol = model.compose_feature(imgs, mods, [""], fabric, device)
                    prob_compose_embedding.append(prob_compose['embedding'].cpu())
                    compose_feature_video_sentence_list.append(compose_feature_video_sentence.cpu())
                    compose_feature_token_list.append(compose_feature_token.cpu())
                    imgs = []
                    mods = []

            # targets feature
            candidate_names, candidate_img = data_loader.dataset.test_name_list, data_loader.dataset.test_img_data
            candidate_features = []
            imgs = []
            for i, img_data in enumerate(tqdm(candidate_img)):
                imgs += [img_data]
                if len(imgs) >= data_loader.batch_size or i == len(candidate_img) - 1:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    tar_feat, tar_img_feat_mm, tar_video_embed, tar_video_token_embed = model.target_fea(imgs, [""], fabric, device)
                    tar_video_embeds.append(tar_video_embed.cpu())
                    tar_video_token_embeds.append(tar_video_token_embed.cpu())
                    imgs = []


            compose_feature_video_sentence_list = torch.cat(compose_feature_video_sentence_list, dim=0)
            compose_feature_token_list = torch.cat(compose_feature_token_list, dim=0)

            prob_compose_embedding = torch.cat(prob_compose_embedding, dim=0)
            tar_video_embeds = torch.cat(tar_video_embeds, dim=0)
            tar_video_token_embeds = torch.cat(tar_video_token_embeds, dim=0)

            model = model.to("cpu")
            # sim_q2t = model.fine_grained_sim(compose_feature_video_sentence_list, compose_feature_frame_sentence_list, compose_feature_token_list, tar_frame_embeds, tar_video_embeds, tar_video_token_embeds).cpu().numpy()
            sims = model.fine_grained_sim(compose_feature_video_sentence_list, compose_feature_token_list, prob_compose_embedding, tar_video_embeds, tar_video_token_embeds, None).cpu().numpy()

            model = model.to(device)
            print("model moved to cpu")


            for i, t in enumerate(test_queries):
                sims[i, candidate_names.index(t['reference_name'])] = -10e10
            sims = -sims
            sorted_inds = np.argsort(sims, axis=-1)
            sorted_ind_names = np.array(candidate_names)[sorted_inds] # (M,N)

            mask = torch.tensor(sorted_ind_names != np.repeat(np.array(reference_name), len(candidate_names)).reshape(len(sorted_ind_names),-1)) # (M,N)
            sorted_ind_names = sorted_ind_names[mask].reshape(sorted_ind_names.shape[0], sorted_ind_names.shape[1] - 1) # (M,N-1)

            subset = np.array(subset) # (M,6)
            subset_mask = (sorted_ind_names[..., None] == subset[:, None, :]).sum(-1).astype(bool) # (M,N-1) label elements in subset
            sorted_subset_names = sorted_ind_names[subset_mask].reshape(sorted_ind_names.shape[0], -1) # (M,6)

            pairid_to_gengeral_pred = {str(int(pair_id)): prediction[:50].tolist()  for pair_id, prediction in zip(pairid, sorted_ind_names)}
            pairid_to_subset_pred = {str(int(pair_id)): prediction[:3].tolist() for pair_id, prediction in zip(pairid, sorted_subset_names)}

            general_submission = {'version': 'rc2', 'metric': 'recall'}
            subset_submission = {'version': 'rc2', 'metric': 'recall_subset'}

            general_submission.update(pairid_to_gengeral_pred)
            subset_submission.update(pairid_to_subset_pred)

            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_dump(general_submission, f"recalls_cirr_{current_time}.json")
            json_dump(subset_submission, f"recalls_cirr_subset_{current_time}.json")


            fabric.barrier()

