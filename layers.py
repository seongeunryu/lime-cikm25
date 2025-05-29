import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
LIME - Candidate-aware Lifetime Attention
'''
class CandidateAware_ClickedNewsAttention(nn.Module):
    def __init__(self, config, news_encoder):
        super(CandidateAware_ClickedNewsAttention, self).__init__()
        self.topic_embedding_dim = config.category_embedding_dim
        self.use_residual_connection = config.use_residual_connection
        self.news_embedding_dim = news_encoder.news_embedding_dim

        self.num_heads = 10  # 4 or 10
        self.head_dim = self.news_embedding_dim // self.num_heads

        assert self.news_embedding_dim % self.num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.last_attn_weights = None
        self.log_weights = False

        self.query_proj = nn.Linear(self.topic_embedding_dim, self.news_embedding_dim)
        self.key_proj = nn.Linear(self.topic_embedding_dim, self.news_embedding_dim)
        self.value_proj = nn.Linear(self.news_embedding_dim, self.news_embedding_dim)

        # attention score scaling
        self.scale = self.news_embedding_dim ** 0.5
        self.dropout = nn.Dropout(p=0.2)

        # Gated Residual
        self.gate_proj = nn.Linear(self.news_embedding_dim, self.news_embedding_dim)
        self.layernorm = nn.LayerNorm(self.news_embedding_dim)

    def initialize(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, clicked_news_embeddings, clicked_news_topic_embeddings, candidate_topic_embeddings, mask=None):
        """
        Input 
        clicked_news_embeddings:        [batch_size, max_history_num, news_embedding_dim]   (V)
        clicked_news_topic_embeddings:  [batch_size, max_history_num, topic_embedding_dim]  (K)
        candidate_topic_embeddings:     [batch_size, news_num, topic_embedding_dim]         (Q)
        mask:                           [batch_size, max_history_num] (optional)
        
        Output
        clicked_news_embeddings:        [batch_size, max_history_num, news_embedding_dim]
        """
        B, H, D = clicked_news_embeddings.size()
        N = candidate_topic_embeddings.size(1)

        Q = self.query_proj(candidate_topic_embeddings).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, d_k]
        K = self.key_proj(clicked_news_topic_embeddings).view(B, H, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, H, d_k]
        V = self.value_proj(clicked_news_embeddings).view(B, H, self.num_heads, self.head_dim).transpose(1, 2)     # [B, heads, H, d_k]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # [B, heads, N, H]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)                     # [B, heads, N, H]
        attn_weights = self.dropout(attn_weights)

        weighted = torch.matmul(attn_weights, V)                          # [B, heads, N, d_k]
        weighted = weighted.transpose(1, 2).contiguous().view(B, N, -1)   # [B, N, D]

        query_weights = F.softmax(torch.norm(Q.transpose(1, 2).reshape(B, N, -1), dim=-1), dim=1)  # [B, N]
        attn_weights_agg = (attn_weights.sum(dim=1) * query_weights.unsqueeze(-1)).sum(dim=1)     # [B, H]
        attn_weights_agg = F.softmax(attn_weights_agg, dim=-1)                                   # [B, H]

        V_orig = clicked_news_embeddings  # [B, H, D]
        weighted_clicked = attn_weights_agg.unsqueeze(-1) * V_orig      # [B, H, D]

        if self.use_residual_connection:
            gate = torch.sigmoid(self.gate_proj(weighted_clicked))     # [B, H, D]
            output = gate * weighted_clicked + (1 - gate) * V_orig     # residual
            output = self.layernorm(output)
        else:
            output = weighted_clicked

        return output, attn_weights_agg
    



class Conv1D(nn.Module):
    def __init__(self, cnn_method, in_channels, cnn_kernel_num, cnn_window_size):
        super(Conv1D, self).__init__()
        assert cnn_method in ['naive', 'group3', 'group5']
        self.cnn_method = cnn_method
        self.in_channels = in_channels
        if self.cnn_method == 'naive':
            self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num, kernel_size=cnn_window_size, padding=(cnn_window_size - 1) // 2)
        elif self.cnn_method == 'group3':
            assert cnn_kernel_num % 3 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=5, padding=2)
        else:
            assert cnn_kernel_num % 5 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=2, padding=0)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=4, padding=1)
            self.conv5 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=5, padding=2)
        self.device = torch.device('cuda')

    # Input
    # feature : [batch_size, feature_dim, length]
    # Output
    # out     : [batch_size, cnn_kernel_num, length]
    def forward(self, feature):
        if self.cnn_method == 'naive':
            return F.relu(self.conv(feature)) # [batch_size, cnn_kernel_num, length]
        elif self.cnn_method == 'group3':
            return F.relu(torch.cat([self.conv1(feature), self.conv2(feature), self.conv3(feature)], dim=1))
        else:
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1], device=self.device)
            return F.relu(torch.cat([self.conv1(feature), \
                                     self.conv2(torch.cat([feature, padding_zeros], dim=1)), \
                                     self.conv3(feature), \
                                     self.conv4(torch.cat([feature, padding_zeros], dim=1)), \
                                     self.conv5(feature)], dim=1))


class Conv2D_Pool(nn.Module):
    def __init__(self, cnn_method, in_channels, cnn_kernel_num, cnn_window_size, last_channel_num):
        super(Conv2D_Pool, self).__init__()
        assert cnn_method in ['naive', 'group3', 'group4']
        self.cnn_method = cnn_method
        self.in_channels = in_channels
        self.last_channel_num = last_channel_num
        self.cnn_window_size = cnn_window_size
        if self.cnn_method == 'naive':
            self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num, kernel_size=[cnn_window_size, last_channel_num], padding=[(cnn_window_size - 1) // 2, 0])
        elif self.cnn_method == 'group3':
            assert cnn_kernel_num % 3 == 0
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=[1, last_channel_num], padding=[0, 0])
            self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=[2, last_channel_num], padding=[0, 0])
            self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=[3, last_channel_num], padding=[1, 0])
        else:
            assert cnn_kernel_num % 4 == 0
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[1, last_channel_num], padding=[0, 0])
            self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[2, last_channel_num], padding=[0, 0])
            self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[3, last_channel_num], padding=[1, 0])
            self.conv4 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[4, last_channel_num], padding=[1, 0])
        self.device = torch.device('cuda')

    # Input
    # feature : [batch_size, feature_dim, length]
    # Output
    # out     : [batch_size, cnn_kernel_num]
    def forward(self, feature):
        length = feature.size(2)
        if self.cnn_method == 'naive':
            conv_relu = F.relu(self.conv(feature), inplace=True)                                                     # [batch_size, cnn_kernel_num, length]
            conv_relu_pool, _ = torch.max(conv_relu[:, :, :length - self.cnn_window_size + 1], dim=2, keepdim=False) # [batch_size, cnn_kernel_num]
            return conv_relu_pool
        elif self.cnn_method == 'group3':
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1, self.last_channel_num], device=self.device)
            conv1_relu = F.relu(self.conv1(feature), inplace=True)
            conv1_relu_pool, _ = torch.max(conv1_relu, dim=2, keepdim=False)
            conv2_relu = F.relu(self.conv2(torch.cat([feature, padding_zeros], dim=2)), inplace=True)
            conv2_relu_pool, _ = torch.max(conv2_relu[:, :, :length - 1], dim=2, keepdim=False)
            conv3_relu = F.relu(self.conv3(feature), inplace=True)
            conv3_relu_pool, _ = torch.max(conv3_relu[:, :, :length - 2], dim=2, keepdim=False)
            return torch.cat([conv1_relu_pool, conv2_relu_pool, conv3_relu_pool], dim=1)
        else:
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1, self.last_channel_num], device=self.device)
            conv1_relu = F.relu(self.conv1(feature), inplace=True)
            conv1_relu_pool, _ = torch.max(conv1_relu, dim=2, keepdim=False)
            conv2_relu = F.relu(self.conv2(torch.cat([feature, padding_zeros], dim=2)), inplace=True)
            conv2_relu_pool, _ = torch.max(conv2_relu[:, :, :length - 1], dim=2, keepdim=False)
            conv3_relu = F.relu(self.conv3(feature), inplace=True)
            conv3_relu_pool, _ = torch.max(conv3_relu[:, :, :length - 2], dim=2, keepdim=False)
            conv4_relu = F.relu(self.conv4(torch.cat([feature, padding_zeros], dim=2)), inplace=True)
            conv4_relu_pool, _ = torch.max(conv4_relu[:, :, :length - 3], dim=2, keepdim=False)
            return torch.cat([conv1_relu_pool, conv2_relu_pool, conv3_relu_pool, conv4_relu_pool], dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, len_q, len_k, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.h = h                  # head_num                  O
        self.d_model = d_model      # word_embedding_dim        O
        self.len_q = len_q          # max_title_length
        self.len_k = len_k          # max_title_length
        self.d_k = d_k              # head_dim                  O
        self.d_v = d_v              # head_dim                  O
        self.out_dim = self.h * self.d_v
        self.attention_scalar = math.sqrt(float(self.d_k))
        self.W_Q = nn.Linear(d_model, self.h*self.d_k, bias=True)
        self.W_K = nn.Linear(d_model, self.h*self.d_k, bias=True)
        self.W_V = nn.Linear(d_model, self.h*self.d_v, bias=True)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)

    # Input
    # Q    : [batch_size, len_q, d_model]
    # K    : [batch_size, len_k, d_model]
    # V    : [batch_size, len_k, d_model]
    # mask : [batch_size, len_k]
    # Output
    # out  : [batch_size, len_q, h * d_v]
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view([batch_size, self.len_q, self.h, self.d_k])                                           # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, self.len_k, self.h, self.d_k])                                           # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, self.len_k, self.h, self.d_v])                                           # [batch_size, len_k, h, d_v]
        Q = Q.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_q, self.d_k])                   # [batch_size * h, len_q, d_k]
        K = K.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_k, self.d_k])                   # [batch_size * h, len_k, d_k]
        V = V.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_k, self.d_v])                   # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.permute(0, 2, 1).contiguous()) / self.attention_scalar                                  # [batch_size * h, len_q, len_k]
        if mask != None:
            _mask = mask.repeat([1, self.h]).view([batch_size * self.h, 1, self.len_k]).repeat([1, self.len_q, 1]) # [batch_size * h, len_q, len_k]
            alpha = F.softmax(A.masked_fill(_mask == 0, -1e9), dim=2)                                              # [batch_size * h, len_q, len_k]
        else:
            alpha = F.softmax(A, dim=2)                                                                            # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, self.len_q, self.d_v])                                 # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view([batch_size, self.len_q, self.out_dim])                  # [batch_size, len_q, h * d_v]
        return out


class ScaledDotProduct_Attention(nn.Module):
    def __init__(self, feature_dim, query_dim, attention_dim):
        super(ScaledDotProduct_Attention, self).__init__()
        self.K = nn.Linear(feature_dim, attention_dim, bias=False)
        self.Q = nn.Linear(query_dim, attention_dim, bias=True)
        self.attention_scalar = math.sqrt(float(attention_dim))

    def initialize(self):
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query, mask=None):
        a = torch.bmm(self.K(feature), self.Q(query).unsqueeze(dim=2)).squeeze(dim=2) / self.attention_scalar # [batch_size, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1)                                          # [batch_size, feature_num]
        else:
            alpha = F.softmax(a, dim=1)                                                                       # [batch_size, feature_num]
        out = torch.bmm(alpha.unsqueeze(dim=1), feature).squeeze(dim=1)                                       # [batch_size, feature_dim]
        return out


class Attention(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(Attention, self).__init__()
        self.affine1 = nn.Linear(feature_dim, attention_dim, bias=True)
        self.affine2 = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, mask=None):
        # 입력된 feature 텐서를 nn.Linear 레이어를 통과시켜 attention_dim 차원으로 변환하고, 
        # 이를 tanh 함수에 적용하여 어텐션 벡터 계산
        attention = torch.tanh(self.affine1(feature))                                 # [batch_size, length, attention_dim]
        # 또 다른 nn.Linear 레이어를 사용하여 어텐션 스코어 값 계산
        a = self.affine2(attention).squeeze(dim=2)                                    # [batch_size, length]
        # 마스크가 주어지면 어텐션 스코어를 마스크 처리하고, 
        # 소프트맥스 함수를 적용하여 어텐션 가중치를 계산
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1) # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)                              # [batch_size, 1, length]
        
        # final: 어텐션 가중치를 사용하여 입력된 특성 텐서를 가중합하여 결과 계산
        out = torch.bmm(alpha, feature).squeeze(dim=1)                                # [batch_size, feature_dim]
        return out

class Attention_with_alpha(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(Attention_with_alpha, self).__init__()
        self.affine1 = nn.Linear(feature_dim, attention_dim, bias=True)
        self.affine2 = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, mask=None):
        # 입력된 feature 텐서를 nn.Linear 레이어를 통과시켜 attention_dim 차원으로 변환하고, 
        # 이를 tanh 함수에 적용하여 어텐션 벡터 계산
        attention = torch.tanh(self.affine1(feature))                                 # [batch_size, length, attention_dim]
        # 또 다른 nn.Linear 레이어를 사용하여 어텐션 스코어 값 계산
        a = self.affine2(attention).squeeze(dim=2)                                    # [batch_size, length]
        # 마스크가 주어지면 어텐션 스코어를 마스크 처리하고, 
        # 소프트맥스 함수를 적용하여 어텐션 가중치를 계산
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1) # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)                              # [batch_size, 1, length]
        # final: 어텐션 가중치를 사용하여 입력된 특성 텐서를 가중합하여 결과 계산
        out = torch.bmm(alpha, feature).squeeze(dim=1)                                # [batch_size, feature_dim]
        return out, alpha

class ScaledDotProduct_CandidateAttention(nn.Module):
    def __init__(self, feature_dim, query_dim, attention_dim):
        super(ScaledDotProduct_CandidateAttention, self).__init__()
        self.K = nn.Linear(feature_dim, attention_dim, bias=False)
        self.Q = nn.Linear(query_dim, attention_dim, bias=True)
        self.attention_scalar = math.sqrt(float(attention_dim))

    def initialize(self):
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query, mask=None):
        a = torch.bmm(self.K(feature), self.Q(query).unsqueeze(dim=2)).squeeze(dim=2) / self.attention_scalar # [batch_size, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1)                                          # [batch_size, feature_num]
        else:
            alpha = F.softmax(a, dim=1)                                                                       # [batch_size, feature_num]
        out = torch.bmm(alpha.unsqueeze(dim=1), feature).squeeze(dim=1)                                       # [batch_size, feature_dim]
        return out

# 주어진 feature와 query 간의 어텐션(attention)을 계산하는 모듈을 정의
# 특정 쿼리(query)에 대해 입력 피쳐(feature) 중에서 중요한 부분에 가중치를 부여 > 중요도에 따라 가중합을 계산
class CandidateAttention(nn.Module):
    def __init__(self, feature_dim, query_dim, attention_dim):
        super(CandidateAttention, self).__init__()
        self.feature_affine = nn.Linear(feature_dim, attention_dim, bias=False)
        self.query_affine = nn.Linear(query_dim, attention_dim, bias=True)
        self.attention_affine = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.feature_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.query_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.query_affine.bias)
        nn.init.xavier_uniform_(self.attention_affine.weight)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query, mask=None):
        a = self.attention_affine(torch.tanh(self.feature_affine(feature) + self.query_affine(query).unsqueeze(dim=1))).squeeze(dim=2) # [batch_size, feature_num]
        if mask is not None:
            # 어텐션 가중치 텐서
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1)                                                                   # [batch_size, feature_num]
        else:
            alpha = F.softmax(a, dim=1)                                                                                                # [batch_size, feature_num]
        # alpha의 모든 배치([batch_size, 1, feature_num])에 대해 각각의 feature 벡터에 어텐션 가중치를 곱한 값을 계산
        # squeeze(dim=1)로 차원 축소([batch_size, feature_dim])
        # 어텐션 가중치를 기반으로 입력 feature와의 가중합을 계산하여 어텐션 결과를 반환
        out = torch.bmm(alpha.unsqueeze(dim=1), feature).squeeze(dim=1)                                                                # [batch_size, feature_dim]
        return out


class MultipleCandidateAttention(nn.Module):
    def __init__(self, feature_dim, query_dim, attention_dim):
        super(MultipleCandidateAttention, self).__init__()
        self.feature_affine = nn.Linear(feature_dim, attention_dim, bias=False)
        self.query_affine = nn.Linear(query_dim, attention_dim, bias=True)
        self.attention_affine = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.feature_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.query_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.query_affine.bias)
        nn.init.xavier_uniform_(self.attention_affine.weight)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_num, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, query_num, feature_dim]
    def forward(self, feature, query, mask=None):
        query_num = query.size(1)
        a = self.attention_affine(torch.tanh(self.feature_affine(feature).unsqueeze(dim=1) + self.query_affine(query).unsqueeze(dim=2))).squeeze(dim=3) # [batch_size, query_num, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask.unsqueeze(dim=1).expand(-1, query_num, -1) == 0, -1e9), dim=2)                                         # [batch_size, query_num, feature_num]
        else:
            alpha = F.softmax(a, dim=2)                                                                                                                 # [batch_size, query_num, feature_num]
        out = torch.bmm(alpha, feature)                                                                                                                 # [batch_size, query_num, feature_dim]
        return out


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, residual=False, layer_norm=False):
        super(GCNLayer, self).__init__()
        self.residual = residual
        self.layer_norm = layer_norm
        if self.residual and in_dim != out_dim:
            raise Exception('To facilitate residual connection, in_dim must equal to out_dim')
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        if self.layer_norm:
            self.layer_normalization = nn.LayerNorm(normalized_shape=[out_dim])

    def initialize(self):
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W.bias)

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = self.W(torch.bmm(graph, feature)) # [batch_size, node_num, feature_num]
        if self.layer_norm:
            out = self.layer_normalization(out) # [batch_size, node_num, feature_num]
        out = F.relu(out)                       # [batch_size, node_num, feature_num]
        if self.residual:
            out = out + feature                 # [batch_size, node_num, feature_num]
        return out

class GCN_(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=0, num_layers=1, dropout=0.1, residual=False, layer_norm=False):
        super(GCN_, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = []
        if self.num_layers == 1:
            self.gcn_layers.append(GCNLayer(in_dim, out_dim, residual=residual, layer_norm=layer_norm))
        else:
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.gcn_layers.append(GCNLayer(in_dim, hidden_dim, residual=residual, layer_norm=layer_norm))
            for i in range(1, self.num_layers - 1):
                self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim, residual=residual, layer_norm=layer_norm))
            self.gcn_layers.append(GCNLayer(hidden_dim, out_dim, residual=residual, layer_norm=layer_norm))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

    def initialize(self):
        for gcn_layer in self.gcn_layers:
            gcn_layer.initialize()

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = feature
        for i in range(self.num_layers - 1):
            out = self.dropout(self.gcn_layers[i](out, graph))
        out = self.gcn_layers[self.num_layers - 1](out, graph)
        return out


class PolyAttention(nn.Module):
    r"""
    Implementation of Poly attention scheme that extracts `K` attention vectors through `K` additive attentions
    """
    def __init__(self, in_embed_dim: int, num_context_codes: int, context_code_dim: int):
        r"""
        Initialization

        Args:
            in_embed_dim: The number of expected features in the input ``embeddings``
            num_context_codes: The number of attention vectors ``K``
            context_code_dim: The number of features in a context code
        """
        super().__init__()
        self.linear = nn.Linear(in_features=in_embed_dim, out_features=context_code_dim, bias=False)
        self.context_codes = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_context_codes, context_code_dim),
                                                                  gain=nn.init.calculate_gain('tanh')))

    def forward(self, embeddings: Tensor, attn_mask: Tensor, bias: Tensor = None):
        r"""
        Forward propagation

        Args:
            embeddings: tensor of shape ``(batch_size, his_length, embed_dim)``
            attn_mask: tensor of shape ``(batch_size, his_length)``
            bias: tensor of shape ``(batch_size, his_length, num_candidates)``

        Returns:
            A tensor of shape ``(batch_size, num_context_codes, embed_dim)``
        """
        proj = torch.tanh(self.linear(embeddings))
        if bias is None:
            weights = torch.matmul(proj, self.context_codes.T)
        else:
            bias = bias.mean(dim=2).unsqueeze(dim=2)
            weights = torch.matmul(proj, self.context_codes.T) + bias
        weights = weights.permute(0, 2, 1)
        weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)
        weights = torch.nn.functional.softmax(weights, dim=2)
        poly_repr = torch.matmul(weights, embeddings)

        return poly_repr

class TargetAwareAttention(nn.Module):
    """Implementation of target-aware attention network"""
    def __init__(self, embed_dim: int):
        r"""
        Initialization

        Args:
            embed_dim: The number of features in query and key vectors
        """
        super().__init__()
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        r"""
        Forward propagation

        Args:
            query: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
            key: tensor of shape ``(batch_size, num_candidates, num_context_codes)``
            value: tensor of shape ``(batch_size, num_candidates, embed_dim)``

        Returns:
            tensor of shape ``(batch_size, num_candidates, embed_dim)``
        """
        proj = torch.nn.functional.gelu(self.linear(query))   # (batch_size, num_context_codes, embed_dim)
        # (batch_size, num_candidates, num_context_codes) key
        # (batch_size, num_context_codes, embed_dim) query
        weights = torch.nn.functional.softmax(torch.matmul(key, proj), dim=2)
        outputs = torch.mul(weights, value)  # (batch_size, num_candidates, num_context_codes)

        return outputs

