# -*- coding: utf-8 -*- 
import os
import torch
from torch import Tensor
import torch.nn as nn
from corpus import Corpus
from dataset import DevTest_Dataset
from torch.utils.data import DataLoader
from evaluate import scoring

'''
Remaining Lifetime-guided Weighting in LIME

[1] RemainingLifetimeWeighting은 현재 잘 설계됨
 - dot product를 base score로 받고,
 - penalty를 적용할지 말지,
 - penalty 계수를 곱할지 말지를 config로 잘 컨트롤하게 되어있음.

[2] self.use_remaining_lifetime_weighting 도 반영되어야 완전한 ablation framework
 - 앞으로 실험 때 True/False 바꿔가면서 remaining lifetime weighting의 효과를 검증 가능
 - 특히, penalty도 True/False로 ablation 가능해서 총 4개의 설정을 쉽게 비교할 수 있음 (w/ weighting, w/o weighting) X (w/ penalty, w/o penalty)

[3] 실험 설계 완성
 - alpha, beta 모두 실험 가능
 - weighting 유무 실험 가능
 - penalty 유무 실험 가능
'''
class RemainingLifetimeWeighting(nn.Module):
    def __init__(self, config):
        super(RemainingLifetimeWeighting, self).__init__()
        self.alpha = config.sigmoid_scaling_alpha   # LANCER의 alpha 계수
        self.beta = config.penalty_scaling_beta
        self.use_expired_penalty = config.use_expired_penalty
        self.use_remaining_lifetime_weighting = config.use_remaining_lifetime_weighting

    def forward(self, user_embedding, news_embedding, remaining_lifetime):
        """
        user_embedding:    [batch_size, news_num, embedding_dim]
        news_embedding:    [batch_size, news_num, embedding_dim]
        remaining_lifetime: [batch_size, news_num]   # 유저-토픽 수명 기준 rtime

        Output: weighted matching score
        """
        # base matching score (dot product)
        base_score = (user_embedding * news_embedding).sum(dim=-1)                  # [batch_size, news_num]

        if not self.use_remaining_lifetime_weighting:
            return base_score


        # Remaining Lifetime-guided Weighting (based on existing LANCER[1] method)
        if self.use_expired_penalty:
            positive_mask = (remaining_lifetime >= 0).float()
            negative_mask = (remaining_lifetime < 0).float()
            weight = torch.sigmoid(self.alpha * remaining_lifetime)
            weight = positive_mask * weight + negative_mask * self.beta * weight
        else:
            # [1] LANCER : A Lifetime-Aware News Recommender System, in AAAI’23
            weight = torch.sigmoid(self.alpha * remaining_lifetime.abs())           # [batch_size, news_num]

        adjusted_score = base_score * weight
        return adjusted_score

    def initialize(self):
        pass  # 필요시 초기화 가능


def pairwise_cosine_similarity(x: Tensor, y: Tensor, zero_diagonal: bool = False) -> Tensor:
    r"""
    Calculates the pairwise cosine similarity matrix

    Args:
        x: tensor of shape ``(batch_size, M, d)``
        y: tensor of shape ``(batch_size, N, d)``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    Returns:
        A tensor of shape ``(batch_size, M, N)``
    """
    x_norm = torch.linalg.norm(x, dim=2, keepdim=True)
    y_norm = torch.linalg.norm(y, dim=2, keepdim=True)
    distance = torch.matmul(torch.div(x, x_norm), torch.div(y, y_norm).permute(0, 2, 1))
    if zero_diagonal:
        assert x.shape[1] == y.shape[1]
        mask = torch.eye(x.shape[1]).repeat(x.shape[0], 1, 1).bool().to(distance.device)
        distance.masked_fill_(mask, 0)

    return distance

def compute_scores(model, corpus, batch_size, mode, result_file, dataset):
    config = model.config
    category_lifetime_map = config.category_lifetime_map
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    dataloader = DataLoader(DevTest_Dataset(corpus, mode), batch_size=batch_size, shuffle=False, num_workers=batch_size // 16, pin_memory=True)
    indices = (corpus.dev_indices if mode == 'dev' else corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            (
                user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity,
                user_freshness, user_user_topic_lifetime, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices,
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity,
                news_freshness, news_user_topic_lifetime
            ) = [x.cuda(non_blocking=True) for x in batch]

            batch_size = user_ID.size(0)
            
            if config.lifetime_type == "fixed":
                remaining_lifetime = config.fixed_lifetime - news_freshness                       # [batch_size, 1 + negative_sample_num]
            elif config.lifetime_type == "topic_wise":
                topic_wise_lifetime = category_lifetime_map[news_category]
                remaining_lifetime = topic_wise_lifetime - news_freshness                         # [batch_size, 1 + negative_sample_num]
            elif config.lifetime_type == "user_topic":
                remaining_lifetime = news_user_topic_lifetime - news_freshness                    # [batch_size, 1 + negative_sample_num]
            else:
                raise ValueError("Invalid lifetime_type")

            scores[index: index+batch_size] = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, \
                                                    user_freshness, user_user_topic_lifetime, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                                                    news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, \
                                                    news_freshness, news_user_topic_lifetime, remaining_lifetime).squeeze(dim=1) # [batch_size]
            index += batch_size
    scores = scores.tolist()
    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, index in enumerate(indices):
        sub_scores[index].append([scores[i], len(sub_scores[index])])
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    if dataset != 'large' or mode != 'test':
        with open(mode + '/ref/truth-%s.txt' % dataset, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg5, ndcg10
    else:
        return None, None, None, None


def get_run_index(result_dir):
    assert os.path.exists(result_dir), 'result directory does not exist'
    max_index = 0
    for result_file in os.listdir(result_dir):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4])
            max_index = max(index, max_index)
    with open(result_dir + '/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1


class AvgMetric:
    def __init__(self, auc, mrr, ndcg5, ndcg10):
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10
        self.avg = (self.auc + self.mrr + (self.ndcg5 + self.ndcg10) / 2) / 3

    def __gt__(self, value):
        return self.avg > value.avg

    def __ge__(self, value):
        return self.avg >= value.avg

    def __lt__(self, value):
        return self.avg < value.avg

    def __le__(self, value):
        return self.avg <= value.avg

    def __str__(self):
        return '%.4f\nAUC = %.4f\nMRR = %.4f\nnDCG@5 = %.4f\nnDCG@10 = %.4f' % (self.avg, self.auc, self.mrr, self.ndcg5, self.ndcg10)
