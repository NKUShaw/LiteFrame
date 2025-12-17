import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from collections import Counter
import torch.nn.functional as F
# import ot

from .text_encoders import TextEmbedding
from .video_encoders import VideoMaeEncoder, R3D18Encoder, R3D50Encoder, C2D50Encoder
from .video_encoders import I3D50Encoder, CSN101Encoder, SLOW50Encoder, EX3DSEncoder
from .video_encoders import EX3DXSEncoder, X3DXSEncoder, X3DSEncoder, X3DMEncoder
from .video_encoders import X3DLEncoder, MVIT16Encoder, MVIT16X4Encoder, MVIT32X3Encoder
from .video_encoders import SLOWFAST50Encoder, SLOWFAST16X8101Encoder
from .image_encoders import VitEncoder, ResnetEncoder, MaeEncoder, SwinEncoder 
from .fusion_module import SumFusion, ConcatFusion, FiLM, GatedFusion 
from .user_encoders import User_Encoder_GRU4Rec, User_Encoder_SASRec, User_Encoder_NextItNet, User_Encoder_NARM, User_Encoder_FMLPRec, User_Encoder_Bert4Rec

class Model(torch.nn.Module):
    def __init__(self, args, pop_prob_list, item_num, bert_model, image_net, video_net, text_content=None):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.item_num = item_num
        self.pop_prob_list = torch.FloatTensor(pop_prob_list)

        if args.model == 'sasrec':
            self.user_encoder = User_Encoder_SASRec(args)
        elif args.model == 'gru4rec':
            self.user_encoder = User_Encoder_GRU4Rec(args)
        elif args.model == 'nextitnet':
            self.user_encoder = User_Encoder_NextItNet(args)
        elif args.model == 'narm':
            self.user_encoder = User_Encoder_NARM(args)
        elif args.model == 'fmlp':
            self.user_encoder = User_Encoder_FMLPRec(args)
        elif args.model == 'bert4rec':
            self.user_encoder = User_Encoder_Bert4Rec(args)

        if 'image' == args.item_tower or 'modal' == args.item_tower:
            if 'resnet' in args.image_model_load:
                self.image_encoder = ResnetEncoder(image_net=image_net, args=args)
            elif 'vit-b-32-clip' in args.image_model_load:
                self.image_encoder = VitEncoder(image_net=image_net, args=args)
            elif 'vit-base-mae' in args.image_model_load:
                self.image_encoder = MaeEncoder(image_net=image_net, args=args)
            elif 'swin_tiny' in args.image_model_load or 'swin_base' in args.image_model_load:
                self.image_encoder = SwinEncoder(image_net=image_net, args=args)

        if 'text' == args.item_tower or 'modal' == args.item_tower:
            self.text_content = torch.LongTensor(text_content)
            self.text_encoder = TextEmbedding(args=args, bert_model=bert_model)
        
        if 'video' == args.item_tower or 'modal' == args.item_tower:
            if 'mae' in args.video_model_load:
                self.video_encoder = VideoMaeEncoder(video_net=video_net, args=args)
            elif 'r3d18' in args.video_model_load:
                self.video_encoder = R3D18Encoder(video_net=video_net, args=args)
            elif 'r3d50' in args.video_model_load:
                self.video_encoder = R3D50Encoder(video_net=video_net, args=args)
            elif 'c2d50' in args.video_model_load:
                self.video_encoder = C2D50Encoder(video_net=video_net, args=args)
            elif 'i3d50' in args.video_model_load:
                self.video_encoder = I3D50Encoder(video_net=video_net, args=args)
            elif 'csn101' in args.video_model_load:
                self.video_encoder = CSN101Encoder(video_net=video_net, args=args)
            elif 'slow50' in args.video_model_load:
                self.video_encoder = SLOW50Encoder(video_net=video_net, args=args)
            elif 'efficient-x3d-s' in args.video_model_load:
                self.video_encoder = EX3DSEncoder(video_net=video_net, args=args)
            elif 'efficient-x3d-xs' in args.video_model_load:
                self.video_encoder = EX3DXSEncoder(video_net=video_net, args=args)
            elif 'x3d-xs' in args.video_model_load:
                self.video_encoder = X3DXSEncoder(video_net=video_net, args=args)
            elif 'x3d-s' in args.video_model_load:
                self.video_encoder = X3DSEncoder(video_net=video_net, args=args)
            elif 'x3d-m' in args.video_model_load:
                self.video_encoder = X3DMEncoder(video_net=video_net, args=args)
            elif 'x3d-l' in args.video_model_load:
                self.video_encoder = X3DLEncoder(video_net=video_net, args=args)
            elif 'mvit-base-16' in args.video_model_load:
                self.video_encoder = MVIT16Encoder(video_net=video_net, args=args)
            elif 'mvit-base-16x4' in args.video_model_load:
                self.video_encoder = MVIT16X4Encoder(video_net=video_net, args=args)
            elif 'mvit-base-32x3' in args.video_model_load:
                self.video_encoder = MVIT32X3Encoder(video_net=video_net, args=args)
            elif 'slowfast-50' in args.video_model_load:
                self.video_encoder = SLOWFAST50Encoder(video_net=video_net, args=args)
            elif 'slowfast16x8-101' in args.video_model_load:
                self.video_encoder = SLOWFAST16X8101Encoder(video_net=video_net, args=args)
            elif 'dinov3' in args.video_model_load:
                self.video_encoder = video_net
            elif 'SV' in args.video_model_load:
                if args.perceiver == 'yes':
                    self.video_encoder = video_net
                else:
                    self.video_encoder = None
                
        self.id_encoder = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_encoder.weight.data)

        self.criterion = nn.CrossEntropyLoss()

        fusion = args.fusion_method.lower()
        if fusion == 'concat' and args.item_tower == 'modal':
            self.fusion_module = ConcatFusion(args=args)

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
        
    # def cosine_cost(self, u, v):
    #     u = F.normalize(u, dim=-1)
    #     v = F.normalize(v, dim=-1)
    #     return 1.0 - torch.matmul(u, v.T)

    # def sinkhorn_plan_pot(self, cost, eps=0.05, n_iter=100):
    #     """
    #     cost: torch.Tensor (bs, bs)
    #     return: torch.Tensor (bs, bs), detached
    #     """
    #     bs = cost.size(0)
    
    #     a = ot.unif(bs)
    #     b = ot.unif(bs)
    
    #     # POT 只能吃 numpy
    #     cost_np = cost.detach().cpu().numpy()
    
    #     P = ot.sinkhorn(a, b, cost_np, reg=eps, numItermax=n_iter)
    
    #     return torch.tensor(P, device=cost.device, dtype=cost.dtype)
        
    # def ot_regularization(self, user, item, eps=0.05):
    #     cost = self.cosine_cost(user, item)   # torch, with grad
    #     P = self.sinkhorn_plan_pot(cost, eps) # detached transport plan
    
    #     ot_loss = torch.sum(P * cost)
    #     return ot_loss

        
    def forward(self, sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask, local_rank, args):
        self.pop_prob_list = self.pop_prob_list.to(local_rank) # Frequency → scaling → normalization → Probability
        debias_logits = torch.log(self.pop_prob_list[sample_items_id.view(-1)]) # torch.Size([batchsize * 11])

        if 'modal' == args.item_tower:
            input_all_text = self.text_encoder(sample_items_text.long())
            input_all_image = self.image_encoder(sample_items_image)
            input_all_video = self.video_encoder(sample_items_video)
            input_embs = self.fusion_module(input_all_text, input_all_image, input_all_video)
        elif 'text' == args.item_tower:
            score_embs = self.text_encoder(sample_items_text.long())
        elif 'image' == args.item_tower:
            score_embs = self.image_encoder(sample_items_image)
        elif 'video' == args.item_tower:
            if "dinov3" == args.video_model_load:
                B, T, F, D = sample_items_video.shape
                # sample_items_video # torch.Size([batchsize, 11, 5, 384]) 
                # 1. non-fine-tuning, just pooling
                # score_embs = sample_items_video.mean(dim=2)  
                # score_embs = score_embs.reshape(-1, 384) # torch.Size([batchsize, 11, 5, 384])
                # 2. perceiver
                if self.video_encoder.query:
                    queries = self.video_encoder.query.expand(B*T, 1, D)
                else:
                    queries = self.video_encoder.query
                score_embs = self.video_encoder(sample_items_video.reshape(B * T, F, D), queries=queries)
                score_embs = score_embs.squeeze(1) 
            elif "SV" == args.video_model_load:
                # print(sample_items_video.shape)
                B, T, F, D = sample_items_video.shape
                if args.perceiver == 'yes':
                    if self.video_encoder.query:
                        queries = self.video_encoder.query.expand(B*T, 1, D)
                    else:
                        queries = self.video_encoder.query
                    
                    score_embs = self.video_encoder(sample_items_video.reshape(B * T, F, D), queries=queries)
                    score_embs = score_embs.squeeze(1) 
                else:
                    score_embs = sample_items_video.mean(dim=2)  
                    score_embs = score_embs.reshape(-1, D) # torch.Size([B*T, D])
                # queries = self.video_encoder.query.expand(B*T, 1, D)
                # score_embs = self.video_encoder(sample_items_video.reshape(B * T, F, D), queries=queries)
                # score_embs = score_embs.squeeze(1) 
            else:
                # The shape of sample_items_video:  torch.Size([batchsize * 11, 5, 3, 224, 224])
                score_embs = self.video_encoder(sample_items_video) # The shape of score_embs:  torch.Size([batchsize * 11, 512])
        elif 'id' == args.item_tower:
            score_embs = self.id_encoder(sample_items_id)

        input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim) # torch.Size([batchsize, 11, 512])        
        if (self.args.model == 'sasrec') or (self.args.model == 'bert4rec'):
            prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank) 
            # Compute the user embedding based on the first 10 videos, torch.Size([batchsize, 10, 512])
        else:
            prec_vec = self.user_encoder(input_embs[:, :-1, :])
        prec_vec = prec_vec.reshape(-1, self.args.embedding_dim)

        ######################################  IN-BATCH CROSS-ENTROPY LOSS  ######################################
        # logits = torch.matmul(F.normalize(prec_vec, dim=-1), F.normalize(score_embs, dim=-1).t()) # (bs * max_seq_len, bs * (max_seq_len + 1))
        # logits = logits / self.args.tau - debias_logits
        logits = torch.matmul(prec_vec, score_embs.t())
        # torch.Size([batchsize * 10, 512]) * torch.Size([512, batchsize * 11]) = torch.Size([batchsize * 10, batchsize * 11])
        logits = logits - debias_logits

        ###################################### MASK USELESS ITEM ######################################
        bs, seq_len = log_mask.size(0), log_mask.size(1)
        label = torch.arange(bs * (seq_len + 1)).reshape(bs, seq_len + 1)
        label = label[:, 1:].to(local_rank).view(-1)

        flatten_item_seq = sample_items_id
        user_history = torch.zeros(bs, seq_len + 2).type_as(sample_items_id)
        user_history[:, :-1] = sample_items_id.view(bs, -1)
        user_history = user_history.unsqueeze(-1).expand(-1, -1, len(flatten_item_seq))
        history_item_mask = (user_history == flatten_item_seq).any(dim=1)
        history_item_mask = history_item_mask.repeat_interleave(seq_len, dim=0)
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)
        
        logits[unused_item_mask] = -1e4
        indices = torch.where(log_mask.view(-1) != 0)
        logits = logits.view(bs * seq_len, -1)
        ce_loss = self.criterion(logits[indices], label[indices])
        
        ###################################### CALCULATE ALIGNMENT AND UNIFORMITY ######################################
        user = prec_vec.view(-1, self.max_seq_len, self.args.embedding_dim)[:, -1, :]
        item = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)[:, -1, :]

        # ot_loss = self.ot_regularization(user, item, eps=0.05)
        
        align = self.alignment(user, item)
        uniform = (self.uniformity(user) + self.uniformity(item)) / 2
        # loss = ce_loss + 0.05 * align
        loss = ce_loss
        return loss, align, uniform
        
    def forward_test(self, sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask, local_rank, args):
        self.pop_prob_list = self.pop_prob_list.to(local_rank)
        debias_logits = torch.log(self.pop_prob_list[sample_items_id.view(-1)]) # torch.Size([batchsize * 11])

        if 'modal' == args.item_tower:
            input_all_text = self.text_encoder(sample_items_text.long())
            input_all_image = self.image_encoder(sample_items_image)
            input_all_video = self.video_encoder(sample_items_video)
            input_embs = self.fusion_module(input_all_text, input_all_image, input_all_video)
        elif 'text' == args.item_tower:
            score_embs = self.text_encoder(sample_items_text.long())
        elif 'image' == args.item_tower:
            score_embs = self.image_encoder(sample_items_image) 
        elif 'video' == args.item_tower:
            # The shape of sample_items_video:  torch.Size([batchsize * 11, 5, 3, 224, 224])
            score_embs = self.video_encoder(sample_items_video) # The shape of score_embs:  torch.Size([batchsize * 11, 512])
        elif 'id' == args.item_tower:
            score_embs = self.id_encoder(sample_items_id)
        input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim) # torch.Size([batchsize, 11, 512])
        if (self.args.model == 'sasrec') or (self.args.model == 'bert4rec'):
            prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank) # torch.Size([batchsize, 10, 512])
        else:
            prec_vec = self.user_encoder(input_embs[:, :-1, :])
        prec_vec = prec_vec.reshape(-1, self.args.embedding_dim)  # torch.Size([batchsize * 10, 512])

        ######################################  IN-BATCH CROSS-ENTROPY LOSS  ######################################
        # logits = torch.matmul(F.normalize(prec_vec, dim=-1), F.normalize(score_embs, dim=-1).t()) # (bs * max_seq_len, bs * (max_seq_len + 1))
        # logits = logits / self.args.tau - debias_logits
        logits = torch.matmul(prec_vec, score_embs.t()) # torch.Size([batchsize * 10, 512]) * torch.Size([512, batchsize * 11]) = torch.Size([batchsize * 10, batchsize * 11])
        logits = logits - debias_logits

        ###################################### MASK USELESS ITEM ######################################
        bs, seq_len = log_mask.size(0), log_mask.size(1) # Shape of log_mask: torch.Size([batchsize, 10])
        label = torch.arange(bs * (seq_len + 1)).reshape(bs, seq_len + 1)
        label = label[:, 1:].to(local_rank).view(-1)

        flatten_item_seq = sample_items_id # Shape of sample_items_id: torch.Size([batchsize * 10])
        user_history = torch.zeros(bs, seq_len + 2).type_as(sample_items_id)
        user_history[:, :-1] = sample_items_id.view(bs, -1)
        user_history = user_history.unsqueeze(-1).expand(-1, -1, len(flatten_item_seq))
        history_item_mask = (user_history == flatten_item_seq).any(dim=1)
        history_item_mask = history_item_mask.repeat_interleave(seq_len, dim=0)
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)
        
        logits[unused_item_mask] = -1e4
        indices = torch.where(log_mask.view(-1) != 0)
        logits = logits.view(bs * seq_len, -1)
        loss = self.criterion(logits[indices], label[indices])

        ###################################### CALCULATE ALIGNMENT AND UNIFORMITY ######################################
        user = prec_vec.view(-1, self.max_seq_len, self.args.embedding_dim)[:, -1, :]
        item = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)[:, -1, :]
        align = self.alignment(user, item)
        uniform = (self.uniformity(user) + self.uniformity(item)) / 2
        
        return loss, align, uniform
