from corpus import Corpus
import time
from config import Config
import torch
import torch.utils.data as data
from numpy.random import randint
from torch.utils.data import DataLoader


class Train_Dataset(data.Dataset):
    def __init__(self, corpus: Corpus):
        self.negative_sample_num = corpus.negative_sample_num
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_abstract_text =  corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_entity = corpus.news_abstract_entity
        if corpus.config.user_encoder == 'SUE':
            self.user_history_graph = corpus.train_user_history_graph
            self.user_history_category_mask = corpus.train_user_history_category_mask
            self.user_history_category_indices = corpus.train_user_history_category_indices
        else:
            self.user_history_graph = None
            self.user_history_category_mask = None
            self.user_history_category_indices = None
        self.train_behaviors = corpus.train_behaviors
        self.max_history_num = corpus.max_history_num
        self.category_num = corpus.config.category_num
        # train_samples.shape : [train_num, 1 + negative_sample_num]
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.train_freshness = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.train_user_topic_lifetime = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.num = len(self.train_behaviors)

    # self.train_samples[i]               = [pos, neg1, neg2, ...]
    # self.train_freshness[i]             = [pos_freshness, neg1_freshness, ...]
    # self.train_user_topic_lifetime[i]   = [pos_lifetime, neg1_lifetime, ...]
    def negative_sampling(self, rank=None):
        print('\n%sBegin negative sampling, training sample num : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
        start_time = time.time()
        
        for i, train_behavior in enumerate(self.train_behaviors):
            pos_index = train_behavior[3]
            neg_indices = train_behavior[4]
            freshness = train_behavior[6]
            pos_lifetime = train_behavior[7]
            neg_lifetimes = train_behavior[8]
            
            self.train_samples[i][0] = pos_index        # positive news index (1)
            self.train_freshness[i][0] = freshness
            self.train_user_topic_lifetime[i][0] = pos_lifetime

            news_num = len(neg_indices)
            used_negative_indices = set()

            for j in range(self.negative_sample_num):
                if news_num <= self.negative_sample_num:
                    k = j % news_num
                else:
                    while True:
                        k = randint(0, news_num - 1)
                        if k not in used_negative_indices:
                            used_negative_indices.add(k)
                            break

                neg_news_index = neg_indices[k]
                neg_lifetime = neg_lifetimes[k]

                self.train_samples[i][j + 1] = neg_news_index
                self.train_freshness[i][j + 1] = freshness
                self.train_user_topic_lifetime[i][j + 1] = neg_lifetime
        end_time = time.time()
        print('%sEnd negative sampling, used time : %.3fs' % ('' if rank is None else ('rank ' + str(rank) + ' : '), end_time - start_time))

    # user_ID                       : [1]
    # user_category                 : [max_history_num]
    # user_subCategory              : [max_history_num]
    # user_title_text               : [max_history_num, max_title_length]
    # user_title_mask               : [max_history_num, max_title_length]
    # user_title_entity             : [max_history_num, max_title_length]
    # user_abstract_text            : [max_history_num, max_abstract_length]
    # user_abstract_mask            : [max_history_num, max_abstract_length]
    # user_abstract_entity          : [max_history_num, max_abstract_length]
    # user_history_mask             : [max_history_num]
    # user_history_graph            : [max_history_num, max_history_num]
    # user_history_category_mask    : [category_num + 1]
    # user_history_category_indices : [max_history_num]
    # user_freshness                : [max_history_num]                                 # for LIME
    # user_user_topic_lifetime      : [max_history_num]                                 # for LIME

    # news_category                 : [1 + negative_sample_num]
    # news_subCategory              : [1 + negative_sample_num]
    # news_title_text               : [1 + negative_sample_num, max_title_length]
    # news_title_mask               : [1 + negative_sample_num, max_title_length]
    # news_title_entity             : [1 + negative_sample_num, max_title_length]
    # news_abstract_text            : [1 + negative_sample_num, max_abstract_length]
    # news_abstract_mask            : [1 + negative_sample_num, max_abstract_length]
    # news_abstract_entity          : [1 + negative_sample_num, max_abstract_length]
    # news_freshness                : [1 + negative_sample_num]                         # for LIME
    # news_user_topic_lifetime      : [1 + negative_sample_num]                         # for LIME

    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = train_behavior[1]                                   # clicked news index
        sample_index = self.train_samples[index]                            # 1 + neg_num candidates index
        news_freshness= self.train_freshness[index]                         # 1 + neg_num candidates freshness
        news_user_topic_lifetime = self.train_user_topic_lifetime[index]    # 1 + neg_num candidates user_topic_lifetime
        behavior_index = train_behavior[5]

        if self.user_history_graph is not None:
            user_history_graph = self.user_history_graph[behavior_index]
            user_history_category_mask = self.user_history_category_mask[behavior_index]
            user_history_category_indices = self.user_history_category_indices[behavior_index]
        else:
            user_history_graph = torch.zeros(self.max_history_num, self.max_history_num, dtype=torch.float)
            user_history_category_mask = torch.zeros(self.category_num + 1, dtype=torch.bool)
            user_history_category_indices = torch.zeros(self.max_history_num, dtype=torch.long)
            
        # freshness (for history)
        freshness_list = train_behavior[9]
        pad_size = self.max_history_num - len(freshness_list)
        user_freshness = freshness_list[-self.max_history_num:] + [0] * max(0, pad_size)
        # user_topic_lifetime (for history)
        user_topic_lifetime_list = train_behavior[10]
        user_user_topic_lifetime = user_topic_lifetime_list[-self.max_history_num:] + [0] * max(0, pad_size)

        return train_behavior[0], \
           self.news_category[history_index], self.news_subCategory[history_index], \
           self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], \
           self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], \
           torch.tensor(user_freshness, dtype=torch.float), \
           torch.tensor(user_user_topic_lifetime, dtype=torch.float), \
           train_behavior[2], user_history_graph, user_history_category_mask, user_history_category_indices, \
           self.news_category[sample_index], self.news_subCategory[sample_index], \
           self.news_title_text[sample_index], self.news_title_mask[sample_index], self.news_title_entity[sample_index], \
           self.news_abstract_text[sample_index], self.news_abstract_mask[sample_index], self.news_abstract_entity[sample_index], \
           torch.tensor(news_freshness, dtype=torch.float), \
           torch.tensor(news_user_topic_lifetime, dtype=torch.float)

    def __len__(self):
        return self.num


class DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: Corpus, mode: str):
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text =  corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.dev_user_history_graph if mode == 'dev' else corpus.test_user_history_graph
        self.user_history_category_mask = corpus.dev_user_history_category_mask if mode == 'dev' else corpus.test_user_history_category_mask
        self.user_history_category_indices = corpus.dev_user_history_category_indices if mode == 'dev' else corpus.test_user_history_category_indices
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
        self.max_history_num = corpus.max_history_num
        self.category_num = corpus.config.category_num
        self.num = len(self.behaviors)

    # user_ID                        : [1]
    # user_category                  : [max_history_num]
    # user_subCategory               : [max_history_num]
    # user_title_text                : [max_history_num, max_title_length]
    # user_title_mask                : [max_history_num, max_title_length]
    # user_title_entity              : [max_history_num, max_title_length]
    # user_abstract_text             : [max_history_num, max_abstract_length]
    # user_abstract_mask             : [max_history_num, max_abstract_length]
    # user_abstract_entity           : [max_history_num, max_abstract_length]
    # user_freshness                 : [max_history_num]                                 # for LIME
    # user_user_topic_lifetime       : [max_history_num]                                 # for LIME
    # user_history_mask              : [max_history_num]
    # user_history_graph             : [max_history_num, max_history_num]
    # user_history_category_mask     : [category_num + 1]
    # user_history_category_indices  : [max_history_num]
    # candidate_news_category        : [1]
    # candidate_news_subCategory     : [1]
    # candidate_news_title_text      : [max_title_length]
    # candidate_news_title_mask      : [max_title_length]
    # candidate_news_title_entity    : [max_title_lenght]
    # candidate_news_abstract_text   : [max_abstract_length]
    # candidate_news_abstract_mask   : [max_abstract_length]
    # candidate_news_abstract_entity : [max_abstract_length]
    # candidate_freshness            : [1]
    # candidate_user_topic_lifetime  : [1]

    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history_index = behavior[1]
        candidate_news_index = behavior[3]          # candidate_news_index
        behavior_index = behavior[4]

        candidate_freshness = behavior[5]
        candidate_user_topic_lifetime = behavior[6]
        freshness_list = behavior[7]
        user_topic_lifetime_list = behavior[8]

        pad_size = self.max_history_num - len(freshness_list)
        user_freshness = freshness_list[-self.max_history_num:] + [0] * max(0, pad_size)
        user_user_topic_lifetime = user_topic_lifetime_list[-self.max_history_num:] + [0] * max(0, pad_size)

        if self.user_history_graph is not None:
            user_history_graph = self.user_history_graph[behavior_index]
            user_history_category_mask = self.user_history_category_mask[behavior_index]
            user_history_category_indices = self.user_history_category_indices[behavior_index]
        else:
            user_history_graph = torch.zeros(self.max_history_num, self.max_history_num, dtype=torch.float)
            user_history_category_mask = torch.zeros(self.category_num + 1, dtype=torch.bool)
            user_history_category_indices = torch.zeros(self.max_history_num, dtype=torch.long)

        return behavior[0], \
            self.news_category[history_index], self.news_subCategory[history_index], \
            self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], \
            self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], \
            torch.tensor(user_freshness, dtype=torch.float), \
            torch.tensor(user_user_topic_lifetime, dtype=torch.float), \
            behavior[2],user_history_graph, user_history_category_mask, user_history_category_indices, \
            self.news_category[candidate_news_index], self.news_subCategory[candidate_news_index], \
            self.news_title_text[candidate_news_index], self.news_title_mask[candidate_news_index], self.news_title_entity[candidate_news_index], \
            self.news_abstract_text[candidate_news_index], self.news_abstract_mask[candidate_news_index], self.news_abstract_entity[candidate_news_index], \
            torch.tensor(candidate_freshness, dtype=torch.float), \
            torch.tensor(candidate_user_topic_lifetime, dtype=torch.float)

    def __len__(self):
        return self.num


if __name__ == '__main__':
    start_time = time.time()
    config = Config()
    dataset_corpus = Corpus(config)
    print('user_num :', len(dataset_corpus.user_ID_dict))
    print('news_num :', len(dataset_corpus.news_title_text))
    print('average title word num :', dataset_corpus.title_word_num / dataset_corpus.news_num)
    print('average abstract word num :', dataset_corpus.abstract_word_num / dataset_corpus.news_num)
    train_dataset = Train_Dataset(dataset_corpus)
    dev_dataset = DevTest_Dataset(dataset_corpus, 'dev')
    test_dataset = DevTest_Dataset(dataset_corpus, 'test')
    train_dataset.negative_sampling()
    end_time = time.time()
    print('load time : %.3fs' % (end_time - start_time))
    print('Train_Dataset :', len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.batch_size // 16)
    for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, \
         user_freshness, user_user_topic_lifetime, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
         news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity, \
         news_freshness, news_user_topic_lifetime) in train_dataloader:
        print('user_ID', user_ID.size(), user_ID.dtype)
        print('user_category', user_category.size(), user_category.dtype)
        print('user_subCategory', user_subCategory.size(), user_subCategory.dtype)
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_title_entity', user_title_entity.size(), user_title_entity.dtype)
        print('user_abstract_text', user_abstract_text.size(), user_abstract_text.dtype)
        print('user_abstract_mask', user_abstract_mask.size(), user_abstract_mask.dtype)
        print('user_abstract_entity', user_abstract_entity.size(), user_abstract_entity.dtype)
        print('user_freshness', user_freshness.size(), user_freshness.dtype)                                                    # for LIME
        print('user_user_topic_lifetime', user_user_topic_lifetime.size(), user_user_topic_lifetime.dtype)                      # for LIME
        print('user_history_mask', user_history_mask.size(), user_history_mask.dtype)
        print('user_history_graph', user_history_graph.size(), user_history_graph.dtype)
        print('user_history_category_mask', user_history_category_mask.size(), user_history_category_mask.dtype)
        print('user_history_category_indices', user_history_category_indices.size(), user_history_category_indices.dtype)
        print('news_category', news_category.size(), news_category.dtype)
        print('news_subCategory', news_subCategory.size(), news_subCategory.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('news_title_entity', news_title_entity.size(), news_title_entity.dtype)
        print('news_abstract_text', news_abstract_text.size(), news_abstract_text.dtype)
        print('news_abstract_mask', news_abstract_mask.size(), news_abstract_mask.dtype)
        print('news_abstract_entity', news_abstract_entity.size(), news_abstract_entity.dtype)
        print('news_freshness', news_freshness.size(), news_freshness.dtype)                                                    # for LIME
        print('news_user_topic_lifetime', news_user_topic_lifetime.size(), news_user_topic_lifetime.dtype)                      # for LIME
        break
    print('Dev_Dataset :', len(dev_dataset))
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, \
         user_freshness, user_user_topic_lifetime, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
         news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity, \
         news_freshness, news_user_topic_lifetime) in dev_dataloader:
        print('user_ID', user_ID.size(), user_ID.dtype)
        print('user_category', user_category.size(), user_category.dtype)
        print('user_subCategory', user_subCategory.size(), user_subCategory.dtype)
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_title_entity', user_title_entity.size(), user_title_entity.dtype)
        print('user_abstract_text', user_abstract_text.size(), user_abstract_text.dtype)
        print('user_abstract_mask', user_abstract_mask.size(), user_abstract_mask.dtype)
        print('user_abstract_entity', user_abstract_entity.size(), user_abstract_entity.dtype)
        print('user_freshness', user_freshness.size(), user_freshness.dtype)                                                    # for LIME
        print('user_user_topic_lifetime', user_user_topic_lifetime.size(), user_user_topic_lifetime.dtype)                      # for LIME
        print('user_history_mask', user_history_mask.size(), user_history_mask.dtype)
        print('user_history_graph', user_history_graph.size(), user_history_graph.dtype)
        print('user_history_category_mask', user_history_category_mask.size(), user_history_category_mask.dtype)
        print('user_history_category_indices', user_history_category_indices.size(), user_history_category_indices.dtype)
        print('news_category', news_category.size(), news_category.dtype)
        print('news_subCategory', news_subCategory.size(), news_subCategory.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('news_title_entity', news_title_entity.size(), news_title_entity.dtype)
        print('news_abstract_text', news_abstract_text.size(), news_abstract_text.dtype)
        print('news_abstract_mask', news_abstract_mask.size(), news_abstract_mask.dtype)
        print('news_abstract_entity', news_abstract_entity.size(), news_abstract_entity.dtype)
        print('news_freshness', news_freshness.size(), news_freshness.dtype)                                                    # for LIME
        print('news_user_topic_lifetime', news_user_topic_lifetime.size(), news_user_topic_lifetime.dtype)                      # for LIME
        break
    print(len(dataset_corpus.dev_indices))
    print('Test_Dataset :', len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, \
         user_freshness, user_user_topic_lifetime, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
         news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity, \
         news_freshness, news_user_topic_lifetime) in test_dataloader:
        print('user_ID', user_ID.size(), user_ID.dtype)
        print('user_category', user_category.size(), user_category.dtype)
        print('user_subCategory', user_subCategory.size(), user_subCategory.dtype)
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_title_entity', user_title_entity.size(), user_title_entity.dtype)
        print('user_abstract_text', user_abstract_text.size(), user_abstract_text.dtype)
        print('user_abstract_mask', user_abstract_mask.size(), user_abstract_mask.dtype)
        print('user_abstract_entity', user_abstract_entity.size(), user_abstract_entity.dtype)
        print('user_freshness', user_freshness.size(), user_freshness.dtype)                                                    # for LIME
        print('user_user_topic_lifetime', user_user_topic_lifetime.size(), user_user_topic_lifetime.dtype)                      # for LIME
        print('user_history_mask', user_history_mask.size(), user_history_mask.dtype)
        print('user_history_graph', user_history_graph.size(), user_history_graph.dtype)
        print('user_history_category_mask', user_history_category_mask.size(), user_history_category_mask.dtype)
        print('user_history_category_indices', user_history_category_indices.size(), user_history_category_indices.dtype)
        print('news_category', news_category.size(), news_category.dtype)
        print('news_subCategory', news_subCategory.size(), news_subCategory.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('news_title_entity', news_title_entity.size(), news_title_entity.dtype)
        print('news_abstract_text', news_abstract_text.size(), news_abstract_text.dtype)
        print('news_abstract_mask', news_abstract_mask.size(), news_abstract_mask.dtype)
        print('news_abstract_entity', news_abstract_entity.size(), news_abstract_entity.dtype)
        print('news_freshness', news_freshness.size(), news_freshness.dtype)                                                    # for LIME
        print('news_user_topic_lifetime', news_user_topic_lifetime.size(), news_user_topic_lifetime.dtype)                      # for LIME
        break
    print(len(dataset_corpus.test_indices))
