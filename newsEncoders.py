# -*- coding: utf-8 -*- 
import pickle
import math
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# import sys
from config import Config
import torch
import torch.nn as nn
from torch import Tensor
print(torch.__version__, torch.version.cuda)
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel
from layers import Conv1D, Conv2D_Pool, MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, CandidateAttention, ScaledDotProduct_Attention



'''
LIME - Lifetime-aware Freshness Encoder

Input:
 - freshness:           [batch_size, news_num] — 초 단위, 예: 6000
 - user_topic_lifetime: [batch_size, news_num] — 초 단위, 예: 86400
Algorithm:
(1) 두 값을 log-scaling + bucketization해서 Embedding
(2) Concat → Dense layer → Tanh

'''
class FreshnessEncoder(nn.Module):
    def __init__(self, config: Config):
        super(FreshnessEncoder, self).__init__()
        embedding_dim=config.freshness_embedding_dim
        hidden_dim=config.lime_hidden_dim
        self.num_buckets = config.num_buckets
        self.freshness_embedding = nn.Embedding(self.num_buckets, embedding_dim)
        self.lifetime_embedding = nn.Embedding(self.num_buckets, embedding_dim)

        self.dense = nn.Linear(embedding_dim * 2, hidden_dim)
        self.activation = nn.Tanh()

    # freshness = 300s (5분) ~ 864000s (10일)처럼 범위가 넓음 → 그대로 Embedding할 수 없어서 log scale로 bucketize
    # 연속형 값을 discretize하여 유연한 표현력을 유지하면서 embedding 가능
    def bucketize(self, x):
        # log-binning + clipping to num_buckets
        x = torch.clamp(x.float(), min=1)  # log(0) 방지
        log_x = torch.log(x)
        scaled = log_x / torch.log(torch.tensor(60 * 60 * 24.0))  # 하루 단위 scaling
        buckets = torch.clamp((scaled * (self.num_buckets / 7)).long(), max=self.num_buckets - 1)
        return buckets

    def forward(self, news_freshness, news_user_topic_lifetime):
        """
        Input
        freshness:              [batch_size, news_num], unit = sec
        user_topic_lifetime:    [batch_size, news_num], unit = sec
        """

        if news_freshness.dim() == 1:
            news_freshness = news_freshness.unsqueeze(1)  # [B] → [B, 1]
        if news_user_topic_lifetime.dim() == 1:
            news_user_topic_lifetime = news_user_topic_lifetime.unsqueeze(1)

        if news_freshness.shape != news_user_topic_lifetime.shape:
            news_user_topic_lifetime = news_user_topic_lifetime.expand_as(news_freshness)
        
        f_bucket = self.bucketize(news_freshness)
        l_bucket = self.bucketize(news_user_topic_lifetime)

        f_embed = self.freshness_embedding(f_bucket)    # [batch_size, news_num, news_embedding_dim]
        l_embed = self.lifetime_embedding(l_bucket)     # [batch_size, news_num, news_embedding_dim]

        concat = torch.cat([f_embed, l_embed], dim=-1)  # [batch_size, news_num, news_embedding_dim * 2]
        output = self.activation(self.dense(concat))    # [batch_size, news_num, hidden_dim]
        return output



class LIME(nn.Module):
    def __init__(self, config: Config, base_news_encoder: nn.Module):
        super(LIME, self).__init__()
        self.final_dim = config.lime_output_dim
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        # self.final_dim=None
        self.base_news_encoder = base_news_encoder  # 기존 뉴스 인코더 (e.g., NAML)
        self.freshness_encoder = FreshnessEncoder(config)
        self.fusion_method = config.fusion_method       # 'concat' or 'add' or 'gated'
        if hasattr(base_news_encoder, "auxiliary_loss"):
            self.auxiliary_loss = base_news_encoder.auxiliary_loss
        else:
            self.auxiliary_loss = None

        # Output dimension 설정
        content_dim = self.base_news_encoder.news_embedding_dim
        freshness_dim = config.lime_hidden_dim

        if self.fusion_method == 'concat':
            self.output_dim = content_dim + freshness_dim
            if self.final_dim:
                self.project = nn.Linear(self.output_dim, self.final_dim)
                self.output_dim = self.final_dim
            else:
                self.project = nn.Identity()
        elif self.fusion_method == 'add':
            assert content_dim == freshness_dim, "Add fusion requires same dim"
            self.output_dim = content_dim
            self.project = nn.Identity()
        elif self.fusion_method == 'gated':
            self.gate = nn.Linear(content_dim + freshness_dim, content_dim)
            self.output_dim = content_dim
            self.project = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        self.news_embedding_dim = self.output_dim

    def initialize(self):
        if hasattr(self.base_news_encoder, 'initialize'):
            self.base_news_encoder.initialize()
        nn.init.xavier_uniform_(self.freshness_encoder.dense.weight)
        nn.init.zeros_(self.freshness_encoder.dense.bias)
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
    
    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        """
        news_freshness, news_user_topic_lifetime: [batch_size, news_num]
        
        """

        content_emb = self.base_news_encoder(title_text, title_mask, title_entity, content_text, content_mask, content_entity, \
                                            category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime)  # [B, N, D_c]
        freshness_emb = self.freshness_encoder(news_freshness, news_user_topic_lifetime)  # [B, N, D_f]
        if freshness_emb.dim() == 2:
            freshness_emb = freshness_emb.unsqueeze(1)

        if self.fusion_method == 'concat':
            combined = torch.cat([content_emb, freshness_emb], dim=-1)
            output = self.project(combined)
        elif self.fusion_method == 'add':
            output = content_emb + freshness_emb
        elif self.fusion_method == 'gated':
            gate_input = torch.cat([content_emb, freshness_emb], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            output = gate * content_emb + (1 - gate) * freshness_emb
        
        return output  # [B, N, output_dim]





class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.word_embedding_dim = config.word_embedding_dim
        self.category_num = config.category_num
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.category_embedding_dim)
        self.category_embedding.weight.requires_grad = False
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.subCategory_embedding_dim)
        self.subCategory_embedding.weight.requires_grad = False
        
        # 240122 - category embedding (pretrained Glove)
        # self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.category_embedding_dim)
        # with open('category_embedding_100_mind.pkl', 'rb') as category_embedding_f:
        #     self.category_embedding.weight.data.copy_(pickle.load(category_embedding_f))
        # self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.subCategory_embedding_dim)
        # with open('subCategory_embedding_100_mind.pkl', 'rb') as subCategory_embedding_f:
        #     self.subCategory_embedding.weight.data.copy_(pickle.load(subCategory_embedding_f))
        
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None
        self.affine = nn.Linear(config.word_embedding_dim, config.word_embedding_dim, bias=True)

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])
        nn.init.xavier_uniform_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    # Input
    # title_text          : [batch_size, news_num, max_title_length]   # [64, 5, 32]
    # title_mask          : [batch_size, news_num, max_title_length]
    # title_entity        : [batch_size, news_num, max_title_length]
    # content_text        : [batch_size, news_num, max_content_length]
    # content_mask        : [batch_size, news_num, max_content_length]
    # content_entity      : [batch_size, news_num, max_content_length]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # user_embedding      : [batch_size, user_embedding_dim]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def feature_fusion(self, news_representation, category, subCategory):
        category_representation = self.category_embedding(category)                                                                                    # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(subCategory)                                                                           # [batch_size, news_num, subCategory_embedding_dim]
        news_representation = torch.cat([news_representation, self.dropout(category_representation), self.dropout(subCategory_representation)], dim=2) # [batch_size, news_num, news_embedding_dim]
        return news_representation

# Our proposed model: CROWN - news encoder
class CROWN(NewsEncoder):
    def __init__(self, config: Config):
        super(CROWN, self).__init__(config)

        self.max_title_length = config.max_title_length
        self.max_body_length = config.max_abstract_length
        self.max_history_num = config.max_history_num
        self.category_embedding_dim = config.category_embedding_dim
        self.intent_embedding_dim = config.intent_embedding_dim
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        # self.news_embedding_dim = config.intent_embedding_dim + config.category_embedding_dim + config.subCategory_embedding_dim
        self.news_embedding_dim = config.intent_embedding_dim * 2 + config.category_embedding_dim + config.subCategory_embedding_dim
        
        # Transformer encoder
        self.title_pos_encoder = PositionalEncoding(config.word_embedding_dim, config.dropout_rate, config.max_title_length)
        self.body_pos_encoder = PositionalEncoding(config.word_embedding_dim, config.dropout_rate, config.max_abstract_length)
        title_encoder_layers = TransformerEncoderLayer(config.word_embedding_dim, config.head_num, config.feedforward_dim, config.dropout_rate, batch_first=True)   # head_num은 word_embedding_dim을 나눌 수 있어야 함
        self.title_transformer = TransformerEncoder(title_encoder_layers, config.num_layers)
        body_encoder_layers = TransformerEncoderLayer(config.word_embedding_dim, config.head_num, config.feedforward_dim, config.dropout_rate, batch_first=True)   # head_num은 word_embedding_dim을 나눌 수 있어야 함
        self.body_transformer = TransformerEncoder(body_encoder_layers, config.num_layers)
        
        # ISAB(Induced Set Attention Block) emcoder - 230915
        self.ISAB = ISAB(dim_in = config.word_embedding_dim, 
                         dim_out = config.word_embedding_dim,
                         num_heads = config.isab_num_heads,        # The number of ISAB heads       4,  choices=[2, 4, 8]
                         num_inds = config.isab_num_inds,          # The number of inducing points  4,  choices=[2, 4, 6, 8]
                         ln = True)
        
        self.category_affine = nn.Linear(config.category_embedding_dim + config.subCategory_embedding_dim, config.category_embedding_dim)
        
        self.intent_num = config.intent_num     # hyperparameter k
        self.alpha = config.alpha
        # self.title_intent_attention = ScaledDotProduct_Attention(config.intent_embedding_dim, config.category_embedding_dim, config.attention_dim)      # 230922
        self.title_intent_attention = Attention(config.intent_embedding_dim, config.attention_dim) 
        self.body_intent_attention = Attention(config.intent_embedding_dim, config.attention_dim)
        self.intent_layers = nn.ModuleList([nn.Linear(config.word_embedding_dim
                                                      + config.category_embedding_dim
                                                      , config.intent_embedding_dim, bias=True) 
                                            for _ in range(self.intent_num)])
        
        self.category_predictor = CategoryPredictor(config.intent_embedding_dim, config.category_num)

    
    def initialize(self):
        super().initialize()
        self.title_intent_attention.initialize()
        self.body_intent_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        # Initialize each intent layer with different weights to learn different embedding for each intent
        for intent_layer in self.intent_layers:
            nn.init.xavier_uniform_(intent_layer.weight)
            nn.init.zeros_(intent_layer.bias)
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

    # Apply k-FC layer for k-intent disentanglement
    def k_intent_disentangle(self, intent_num, news_embedding):                                        
        k_intent_embeddings = []
        for i in range(intent_num):
            # Apply different linear transformations for each intent
            intent_embedding = F.relu(self.intent_layers[i](news_embedding), inplace=True)      # [batch_size * news_num, intent_embedding_dim]     
            # Expand the dimension (axis 1)
            intent_embedding_exp = intent_embedding.unsqueeze(1)
            k_intent_embeddings.append(intent_embedding_exp)
        # Concatenate the k_intent_embeddings along the second axis (axis 1)
        k_intent_embeddings = torch.cat(k_intent_embeddings, dim=1)                             # [batch_size * news_num, intent_length, intent_embedding_dim]

        return k_intent_embeddings

    def similarity_compute(self, title, body):                              # [batch_size * news_num, intent_embedding_dim]
        cosine_similarity = F.cosine_similarity(title, body, dim=1)             
        title_body_similarity = (cosine_similarity + 1) / 2.0
        return title_body_similarity    
    
    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        
        t_mask = title_mask.view([batch_news_num, self.max_title_length])                                   # [batch_size * news_num, max_title_length]
        b_mask = content_mask.view([batch_news_num, self.max_body_length])                                  # [batch_size * news_num, max_body_length]
        
        # Word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])          # [batch_size * news_num, max_title_length, word_embedding_dim]
        body_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_body_length, self.word_embedding_dim])          # [batch_size * news_num, max_content_length, word_embedding_dim]
        
        # Transformer encoding
        title_p = self.title_pos_encoder(title_w)                                                       # [batch_size * news_num, max_title_length, news_embedding_dim]
        title_t = self.title_transformer(title_p)                                                       # [batch_size * news_num, max_title_length, news_embedding_dim]
        title_embedding = title_t.mean(dim=1).view([batch_size * news_num, self.word_embedding_dim])    # [batch_size * news_num, news_embedding_dim]    
        
        body_p = self.body_pos_encoder(body_w)                                                          # [batch_size * news_num, max_content_length, news_embedding_dim]
        body_t = self.body_transformer(body_p)                                                          # [batch_size * news_num, max_content_length, news_embedding_dim]
        body_embedding = body_t.mean(dim=1).view([batch_size * news_num, self.word_embedding_dim])      # [batch_size * news_num, news_embedding_dim]   
        
        
        # ISAB(Induced Set Attention Block) encoding
        # title_isab1 = self.dropout(self.ISAB(title_w))                                                      # [batch_size * news_num, max_title_length, news_embedding_dim]
        # title_isab2 = self.dropout(self.ISAB(title_isab1))                                                  # [batch_size * news_num, max_title_length, news_embedding_dim]
        # title_embedding = title_isab2.mean(dim=1).view([batch_news_num, self.word_embedding_dim])           # [batch_size, news_num, news_embedding_dim]
        # # title_embedding = title_isab1.mean(dim=1).view([batch_news_num, self.word_embedding_dim])         
        
        # body_isab1 = self.dropout(self.ISAB(body_w))                                                        # [batch_size * news_num, max_title_length, news_embedding_dim]
        # body_isab2 = self.dropout(self.ISAB(body_isab1))                                                    # [batch_size * news_num, max_title_length, news_embedding_dim]
        # body_embedding = body_isab2.mean(dim=1).view([batch_news_num, self.word_embedding_dim])             # [batch_size, news_num, news_embedding_dim]
        # # body_embedding = body_isab1.mean(dim=1).view([batch_news_num, self.word_embedding_dim])          
        
        # Word-level Attention encoding (* instead of title_t.mean(dim=1))       
        # title_embedding = self.title_attention(title_t, mask=t_mask).view([batch_size, news_num, self.news_embedding_dim])                  # [batch_size, news_num, news_embedding_dim]  
        # body_embedding = self.body_attention(body_t, mask=b_mask).view([batch_size, news_num, self.news_embedding_dim])                     # [batch_size, news_num, news_embedding_dim]  
        
        # Category-aware intent disentanglement
        category_representation = self.category_affine(torch.cat([self.category_embedding(category),
                                                                  self.subCategory_embedding(subCategory)], 
                                                                  dim=2)).view([batch_news_num, self.category_embedding_dim])   # [batch_size * news_num, category_embedding_dim] 
        category_aware_title_embedding = torch.cat([title_embedding, category_representation], dim=1)                            # [batch_size * news_num, news_embedding_dim + category_embedding_dim]
        category_aware_body_embedding = torch.cat([body_embedding, category_representation], dim=1)                             # [batch_size * news_num, news_embedding_dim + category_embedding_dim]
        # for no category
        # category_aware_title_embedding=  title_embedding                                                                         # [batch_size * news_num, news_embedding_dim]
        # category_aware_body_embedding =  body_embedding                                                                          # [batch_size * news_num, news_embedding_dim]


        k = self.intent_num
        title_k_intent_embeddings = self.k_intent_disentangle(k, category_aware_title_embedding)              # [batch_size * news_num, intent_length(k), intent_embedding_dim]
        body_k_intent_embeddings = self.k_intent_disentangle(k, category_aware_body_embedding)                # [batch_size * news_num, intent_length(k), intent_embedding_dim]
        
        # Intent-based Attention
        title_intent_embedding = self.title_intent_attention(title_k_intent_embeddings)              # [batch_size * news_num, intent_embedding_dim]  [batch_size * news_num, 1, intent_length(k)]
        body_intent_embedding = self.body_intent_attention(body_k_intent_embeddings)                  # [batch_size * news_num, intent_embedding_dim]  [batch_size * news_num, 1, intent_length(k)]
        
        # Category predictor
        # title_embedding     : [batch_size * news_num, intent_embedding_dim]
        # target category     : [batch_size * news_num, 1]
        target_category = category.view([batch_news_num, 1])
        category_loss = self.category_predictor(title_intent_embedding, target_category, self.category_num)
        
        self.auxiliary_loss = category_loss * self.alpha
        
        # Title-Body similarity computation
        title_body_similarity = self.similarity_compute(title_intent_embedding, body_intent_embedding).view([batch_size * news_num, 1])  # [batch_size * news_num, 1]
        
        news_representation = torch.cat([title_intent_embedding, title_body_similarity * body_intent_embedding], dim=1).view([batch_size, news_num, self.intent_embedding_dim * 2])       # [batch_size, news_num, intent_embedding_dim * 2]         # concat 
        # news_representation = (title_intent_embedding + title_body_similarity * body_intent_embedding).view([batch_size, news_num, self.intent_embedding_dim])                          # [batch_size, news_num, title(body) intent_embedding_dim]     # average(weighted sum)
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                   # [batch_size, news_num, intent_embedding_dim + 100]

        return news_representation                                                                                              # [batch_size, news_num, news_embedding_dim]    

class CategoryPredictor(nn.Module):
    def __init__(self, title_embedding, category_num):
        super(CategoryPredictor, self).__init__()
        self.fc = nn.Linear(title_embedding, category_num)

    def one_hot_encode(self, target, category_num):
        batch_size, _ = target.size()
        one_hot = torch.zeros(batch_size, category_num, device=target.device)
        target = target.long()
        one_hot.scatter_(1, target, 1)
        return one_hot
    # Input: title_intent_embedding             # [batch_size * news_num, intent_embedding_dim]
    # Output: category loss (auxiliary loss)    # [batch_size, news_num]
    def forward(self, title_intent_embedding, targets, category_num):
        category_logits = self.fc(title_intent_embedding)               # [batch_size * news_num, category_num]
        one_hot_targets = self.one_hot_encode(targets, category_num)    # [batch_size * news_num, category_num]
        category_loss = F.cross_entropy(category_logits, one_hot_targets)

        return category_loss

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


# Collaborative News Encoding(CNE)
    # cross-selective encoding + cross-attentive encoding
class CNE(NewsEncoder):
    def __init__(self, config: Config):
        super(CNE, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.news_embedding_dim = config.hidden_dim * 4 + config.category_embedding_dim + config.subCategory_embedding_dim # 900
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        # selective LSTM encoder
        self.title_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.content_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.title_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.title_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        self.content_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.content_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        # self-attention
        self.title_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        self.content_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        # cross-attention
        self.title_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)
        self.content_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)

    def initialize(self):
        super().initialize()
        for parameter in self.title_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        for parameter in self.content_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        nn.init.xavier_uniform_(self.title_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.title_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.title_M.bias)
        nn.init.xavier_uniform_(self.content_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.content_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.content_M.bias)
        self.title_self_attention.initialize()
        self.content_self_attention.initialize()
        self.title_cross_attention.initialize()
        self.content_cross_attention.initialize()
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        title_mask = title_mask.view([batch_news_num, self.max_title_length])                                                                              # [batch_size * news_num, max_title_length]
        content_mask = content_mask.view([batch_news_num, self.max_content_length])                                                                        # [batch_size * news_num, max_content_length]
        title_mask[:, 0] = 1   # To avoid empty input of LSTM
        content_mask[:, 0] = 1 # To avoid empty input of LSTM
        title_length = title_mask.sum(dim=1, keepdim=False).long()                                                                                         # [batch_size * news_num]
        content_length = content_mask.sum(dim=1, keepdim=False).long()                                                                                     # [batch_size * news_num]
        sorted_title_length, sorted_title_indices = torch.sort(title_length, descending=True)                                                              # [batch_size * news_num]
        _, desorted_title_indices = torch.sort(sorted_title_indices, descending=False)                                                                     # [batch_size * news_num]
        sorted_content_length, sorted_content_indices = torch.sort(content_length, descending=True)                                                        # [batch_size * news_num]
        _, desorted_content_indices = torch.sort(sorted_content_indices, descending=False)                                                                 # [batch_size * news_num]
        # 1. word embedding
        title = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])                       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])                 # [batch_size * news_num, max_content_length, word_embedding_dim]
        sorted_title = pack_padded_sequence(title.index_select(0, sorted_title_indices), sorted_title_length.cpu(), batch_first=True)                      # [batch_size * news_num, max_title_length, word_embedding_dim]
        sorted_content = pack_padded_sequence(content.index_select(0, sorted_content_indices), sorted_content_length.cpu(), batch_first=True)              # [batch_size * news_num, max_content_length, word_embedding_dim]
        # [Cross-selective Encoding]
        # 2. selective LSTM encoding
        # parallel bidirectional LSTMs (1),(2)
        # h: hidden state, c: cell state
        sorted_title_h, (sorted_title_h_n, sorted_title_c_n) = self.title_lstm(sorted_title)
        sorted_content_h, (sorted_content_h_n, sorted_content_c_n) = self.content_lstm(sorted_content)
        # semantic memory vector (3)
        sorted_title_m = torch.cat([sorted_title_c_n[0], sorted_title_c_n[1]], dim=1)                                                                      # [batch_size * news_num, hidden_dim * 2]
        sorted_content_m = torch.cat([sorted_content_c_n[0], sorted_content_c_n[1]], dim=1)                                                                # [batch_size * news_num, hidden_dim * 2]
        sorted_title_h, _ = pad_packed_sequence(sorted_title_h, batch_first=True, total_length=self.max_title_length)                                      # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_h, _ = pad_packed_sequence(sorted_content_h, batch_first=True, total_length=self.max_content_length)                                # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # sigmoid gate function (4),(5)
        sorted_title_gate = torch.sigmoid(self.title_H(sorted_title_h) + self.title_M(sorted_content_m).unsqueeze(dim=1))                                  # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_gate = torch.sigmoid(self.content_H(sorted_content_h) + self.content_M(sorted_title_m).unsqueeze(dim=1))                            # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # cross selective feature (final output of cross-selective encoding)
        title_h = (sorted_title_h * sorted_title_gate).index_select(0, desorted_title_indices)                                                             # [batch_size * news_num, max_title_length, hidden_dim * 2]
        content_h = (sorted_content_h * sorted_content_gate).index_select(0, desorted_content_indices)                                                     # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # [Cross-attentive Encoding]
        # 3. self-attention (6)
        title_self = self.title_self_attention(title_h, title_mask)                                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_self = self.content_self_attention(content_h, content_mask)                                                                                # [batch_size * news_num, hidden_dim * 2]
        # 4. cross-attention (7),(8)
        title_cross = self.title_cross_attention(title_h, content_self, title_mask)                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_cross = self.content_cross_attention(content_h, title_self, content_mask)                                                                  # [batch_size * news_num, hidden_dim * 2]
        news_representation = torch.cat([title_self + title_cross, content_self + content_cross], dim=1).view([batch_size, news_num, self.hidden_dim * 4]) # [batch_size, news_num, hidden_dim * 4] 800
        # 5. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                                              # [batch_size, news_num, news_embedding_dim] 900
        return news_representation


class CNN(NewsEncoder):
    def __init__(self, config: Config):
        super(CNN, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        
    def initialize(self):
        super().initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))                                                           # [batch_size * news_num, max_sentence_length, cnn_kernel_num]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.cnn_kernel_num])                        # [batch_size, news_num, cnn_kernel_num]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class MHSA(NewsEncoder):
    def __init__(self, config: Config):
        super(MHSA, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.feature_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)
        self.attention = Attention(config.head_num*config.head_dim, config.attention_dim)
        self.news_embedding_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        
    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. multi-head self-attention
        c = self.dropout(self.multiheadAttention(w, w, w, mask))                                                                    # [batch_size * news_num, max_sentence_length, news_embedding_dim]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.feature_dim])                           # [batch_size, news_num, news_embedding_dim]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class KCNN(NewsEncoder):
    def __init__(self, config: Config):
        super(KCNN, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.entity_embedding_dim = config.entity_embedding_dim
        self.context_embedding_dim = config.context_embedding_dim
        self.entity_embedding = nn.Embedding(num_embeddings=config.entity_size, embedding_dim=self.entity_embedding_dim)
        self.context_embedding = nn.Embedding(num_embeddings=config.entity_size, embedding_dim=self.context_embedding_dim)
        with open('entity_embedding-%s.pkl' % config.dataset, 'rb') as entity_embedding_f:
            self.entity_embedding.weight.data.copy_(pickle.load(entity_embedding_f))
        with open('context_embedding-%s.pkl' % config.dataset, 'rb') as context_embedding_f:
            self.context_embedding.weight.data.copy_(pickle.load(context_embedding_f))
        self.M_entity = nn.Linear(self.entity_embedding_dim, self.word_embedding_dim, bias=True)
        self.M_context = nn.Linear(self.context_embedding_dim, self.word_embedding_dim, bias=True)
        self.knowledge_cnn = Conv2D_Pool(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size, 3)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        
    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.M_entity.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.M_entity.bias)
        nn.init.xavier_uniform_(self.M_context.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.M_context.bias)
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word & entity & context embedding
        word_embedding = self.word_embedding(title_text).view([batch_news_num, self.max_title_length, self.word_embedding_dim])                                  # [batch_size * news_num, max_title_length, word_embedding_dim]
        entity_embedding = self.entity_embedding(title_entity).view([batch_news_num, self.max_title_length, self.entity_embedding_dim])                          # [batch_size * news_num, max_title_length, entity_embedding_dim]
        context_embedding = self.context_embedding(title_entity).view([batch_news_num, self.max_title_length, self.context_embedding_dim])                       # [batch_size * news_num, max_title_length, context_embedding_dim]
        W = torch.stack([word_embedding, torch.tanh(self.M_entity(entity_embedding)), torch.tanh(self.M_context(context_embedding))], dim=3).permute(0, 2, 1, 3) # [batch_size * news_num, word_embedding_dim, max_title_length, 3]
        # 2. knowledge-aware CNN
        news_representation = self.knowledge_cnn(W).view([batch_size, news_num, self.cnn_kernel_num])                                                            # [batch_size, news_num, cnn_kernel_num]
        # 3. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                                                    # [batch_size, news_num, news_embedding_dim]
        return news_representation


class NAML(NewsEncoder):
    def __init__(self, config: Config):
        super(NAML, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.news_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.subCategory_affine = nn.Linear(config.subCategory_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        
    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.subCategory_affine.weight)
        nn.init.zeros_(self.subCategory_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim]) # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. CNN encoding
        
        # print(f"Conv1D input shape: {title_w.permute(0, 2, 1).shape}")
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))
        # print(f"Conv1D output shape: {title_c.shape}")                                                 # [batch_size * news_num, max_title_length, cnn_kernel_num]
        
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                            # [batch_size * news_num, max_content_length, cnn_kernel_num]
        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, news_num, self.cnn_kernel_num])                               # [batch_size, news_num, cnn_kernel_num]
        content_representation = self.content_attention(content_c).view([batch_size, news_num, self.cnn_kernel_num])                         # [batch_size, news_num, cnn_kernel_num]
        # 4. category and subCategory encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                              # [batch_size, news_num, cnn_kernel_num]
        subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)), inplace=True)                  # [batch_size, news_num, cnn_kernel_num]
        # 5. multi-view attention
        feature = torch.stack([title_representation, content_representation, category_representation, subCategory_representation], dim=2)    # [batch_size, news_num, 4, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                            # [batch_size, news_num, 4, 1]
        news_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                    # [batch_size, news_num, cnn_kernel_num]
        return news_representation
 
# PNE(Personalized News Encoder): NPA - news encoder
# - module 1: word embedding
# - module 2: convolutional neural network(CNN)
# - module 3: word-level personalized attention network

class PNE(NewsEncoder):
    def __init__(self, config: Config):
        super(PNE, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.personalized_embedding_dim = config.personalized_embedding_dim
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.dense = nn.Linear(config.user_embedding_dim, config.personalized_embedding_dim, bias=True)
        self.personalizedAttention = CandidateAttention(config.cnn_kernel_num, config.personalized_embedding_dim, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        
    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense.bias)
        self.personalizedAttention.initialize()
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))                                                           # [batch_size * news_num, max_sentence_length, cnn_kernel_num]
        # 3. attention layer
        q_w = F.relu(self.dense(user_embedding), inplace=True).repeat([news_num, 1])                                                # [batch_size * news_num, personalized_embedding_dim]
        news_representation = self.personalizedAttention(c, q_w, mask).view([batch_size, news_num, self.cnn_kernel_num])            # [batch_size, news_num, cnn_kernel_num]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation

class HDC(NewsEncoder):
    def __init__(self, config: Config):
        super(HDC, self).__init__(config)
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.word_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.word_embedding_dim)
        self.HDC_sequence_length = config.max_title_length + 2
        self.HDC_filter_num = config.HDC_filter_num
        self.dilated_conv1 = nn.Conv1d(in_channels=config.word_embedding_dim, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2, dilation=1)
        self.dilated_conv2 = nn.Conv1d(in_channels=self.HDC_filter_num, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2 + 1, dilation=2)
        self.dilated_conv3 = nn.Conv1d(in_channels=self.HDC_filter_num, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2 + 2, dilation=3)
        self.layer_norm1 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.layer_norm2 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.layer_norm3 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.news_embedding_dim = None
        
    def initialize(self):
        super().initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. sequence embeddings
        word_embedding = self.word_embedding(title_text).permute(0, 1, 3, 2)                                                 # [batch_size, news_num, word_embedding_dim, title_length]
        category_embedding = self.category_embedding(category).unsqueeze(dim=3)                                              # [batch_size, news_num, word_embedding_dim, 1]
        subCategory_embedding = self.subCategory_embedding(subCategory).unsqueeze(dim=3)                                     # [batch_size, news_num, word_embedding_dim, 1]
        d0 = torch.cat([category_embedding, subCategory_embedding, word_embedding], dim=3)                                   # [batch_size, news_num, word_embedding_dim, HDC_sequence_length]
        d0 = d0.view([batch_news_num, self.word_embedding_dim, self.HDC_sequence_length])                                    # [batch_size * news_num, word_embedding_dim, HDC_sequence_length]
        # 2. hierarchical dilated convolution
        d1 = F.relu(self.layer_norm1(self.dilated_conv1(d0)), inplace=True)                                                  # [batch_size * news_num, HDC_filter_num, HDC_sequence_length]
        d2 = F.relu(self.layer_norm2(self.dilated_conv2(d1)), inplace=True)                                                  # [batch_size * news_num, HDC_filter_num, HDC_sequence_length]
        d3 = F.relu(self.layer_norm3(self.dilated_conv3(d2)), inplace=True)                                                  # [batch_size * news_num, HDC_filter_num, HDC_sequence_length]
        d0 = d0.view([batch_size, news_num, self.word_embedding_dim, self.HDC_sequence_length])                              # [batch_size, news_num, word_embedding_dim, HDC_sequence_length]
        dL = torch.stack([d1, d2, d3], dim=1).view([batch_size, news_num, 3, self.HDC_filter_num, self.HDC_sequence_length]) # [batch_size, news_num, 3, HDC_filter_num, HDC_sequence_length]
        return (d0, dL)

class DAE(NewsEncoder):
    def __init__(self, config: Config):
        super(DAE, self).__init__(config)
        self.Alpha = config.Alpha
        assert self.Alpha > 0, 'Reconstruction loss weight must be greater than 0'
        self.f1 = nn.Linear(config.word_embedding_dim, config.hidden_dim, bias=True)
        self.f2 = nn.Linear(config.hidden_dim, config.word_embedding_dim, bias=True)
        self.news_embedding_dim = config.hidden_dim + config.category_embedding_dim + config.subCategory_embedding_dim
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
        
    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.f1.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.f1.bias)
        nn.init.xavier_uniform_(self.f2.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.f2.bias)
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        title_mask = title_mask.unsqueeze(dim=3)
        content_mask = content_mask.unsqueeze(dim=3)
        word_embedding = torch.sigmoid(((self.word_embedding(title_text) * title_mask).sum(dim=2) + (self.word_embedding(content_text) * content_mask).sum(dim=2)) \
                         / (title_mask.sum(dim=2, keepdim=False) + content_mask.sum(dim=2, keepdim=False)))           # [batch_size, news_num, word_embedding_dim]
        corrupted_word_embedding = self.dropout_(word_embedding)                                                      # [batch_size, news_num, word_embedding_dim]
        news_representation = torch.sigmoid(self.f1(corrupted_word_embedding))                                        # [batch_size, news_num, news_embedding_dim]
        denoised_word_embedding = torch.sigmoid(self.f2(news_representation))                                         # [batch_size, news_num, word_embedding_dim]
        self.auxiliary_loss = torch.norm(word_embedding - denoised_word_embedding, dim=2, keepdim=False) * self.Alpha # [batch_size, news_num]
        # feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                         # [batch_size, news_num, news_embedding_dim]
        return news_representation


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


'''
class PLM_BERT(NewsEncoder, RobertaPreTrainedModel):
    def __init__(self, config: Config, plm_config: RobertaConfig):
        super(PLM_BERT, self).__init__(config, plm_config)
        
        self.roberta = RobertaModel(plm_config)
        if config.freeze_transformer:
            for param in self.roberta.parameters():
                param.requires_grad = False
                
        self.apply_reduce_dim = config.apply_reduce_dim # True or False
        
        if self.apply_reduce_dim:
            assert config.word_embedding_dim is not None
            self.reduce_dim = nn.Linear(in_features=plm_config.hidden_size, out_features=config.word_embedding_dim)
            self._embed_dim = config.word_embedding_dim
        else:
            self._embed_dim = plm_config.hidden_size
        
        self.use_content = config.use_content
        if self.use_content:
            self.combine_type = config.combine_type
            if self.combine_type == 'linear':
                self.linear_combine = nn.Linear(in_features=self._embed_dim * 2, out_features=self._embed_dim)
            elif self.combine_type == 'lstm':
                self.lstm = nn.LSTM(input_size=self._embed_dim * 2, hidden_size=self._embed_dim // 2,
                                    num_layers= config.lstm_num_layers, batch_first=True, dropout=config.dropout_rate,
                                    bidirectional=True)
                self._embed_dim = (self._embed_dim // 2) * 2
        
        self.max_title_length = config.max_title_length
        self.max_abstract_length = config.max_abstract_length
        
        self.init_weights()
        

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.reduce_dim.weight)
        nn.init.xavier_uniform_(self.linear_combine.weight)
        

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, news_freshness, news_user_topic_lifetime):
        
        r"""
        Forward propagation

        Args:
            title_encoding: tensor of shape ``(batch_size, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, title_length)``.
            sapo_encoding: tensor of shape ``(batch_size, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, sapo_length)``.

        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        
        """
        Input
        title_text          : [batch_size, news_num, max_title_length]   # [64, 5, 32]
        title_mask          : [batch_size, news_num, max_title_length]
        title_entity        : [batch_size, news_num, max_title_length]
        content_text        : [batch_size, news_num, max_content_length]
        content_mask        : [batch_size, news_num, max_content_length]
        content_entity      : [batch_size, news_num, max_content_length]
        category            : [batch_size, news_num]
        subCategory         : [batch_size, news_num]
        user_embedding      : [batch_size, user_embedding_dim]
        Output
        news_representation : [batch_size, news_num, news_embedding_dim]
        """
        
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        
        title_text = title_text.reshape(batch_news_num, self.max_title_length)
        title_mask = title_mask.reshape(batch_news_num, self.max_title_length)
        content_text = content_text.reshape(batch_news_num, self.max_abstract_length)
        content_mask = content_mask.reshape(batch_news_num, self.max_abstract_length) 
        
        news_info = []
        # Title encoder
        title_word_embed = self.roberta(input_ids=title_text, attention_mask=title_mask)[0]
        print('title_word_embed shape: ', title_word_embed.shape)
        title_repr = title_word_embed[:, 0, :]
        if self.apply_reduce_dim:
            title_repr = self.reduce_dim(title_repr)
            title_repr = self.dropout(title_repr)
        news_info.append(title_repr)

        # Content encoder
        if self.use_content:
            content_word_embed = self.roberta(input_ids=content_text, attention_mask=content_mask)[0]
            content_repr = content_word_embed[:, 0, :]
            
            if self.apply_reduce_dim:
                content_repr = self.reduce_dim(content_repr)
                content_repr = self.dropout(content_repr)
            news_info.append(content_repr)

            if self.combine_type == 'linear':
                news_info = torch.cat(news_info, dim=1)
                news_representation = self.linear_combine(news_info).view([batch_size, news_num, self.news_embedding_dim])
                return news_representation
            
            elif self.combine_type == 'lstm':
                news_info = torch.cat(news_info, dim=1)
                news_representation, _ = self.lstm(news_info)
                news_representation = news_representation.reshape(batch_size, news_num, self.news_embedding_dim)
                return news_representation
        else:
            news_representation = title_repr.reshape(batch_size, news_num, self.news_embedding_dim)
            return news_representation
'''