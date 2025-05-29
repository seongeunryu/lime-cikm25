#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import gc
import shutil
from config import Config
import torch
from corpus import Corpus
from model import Model
from trainer import Trainer, distributed_train
from util import compute_scores, get_run_index
import torch.multiprocessing as mp
from transformers import AutoTokenizer, RobertaConfig

# function: model training
def train(config, corpus):
    model = Model(config) # NewsRec model object configuration
    model.initialize() # model initialization
    run_index = get_run_index(config.result_dir)
    # Parallel processing, if distributed training is possible (python 3.7)
    if config.world_size == 1:
        trainer = Trainer(model, config, corpus, run_index)
        trainer.train()
        trainer = None
        del trainer
    else:
        try:
            mp.spawn(distributed_train, args=(model, config, corpus, run_index), nprocs=config.world_size, join=True)
        except Exception as e:
            print(e)
            e = str(e).lower()
            if 'cuda' in e or 'pytorch' in e:
                exit()
    config.run_index = run_index
    model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()


def dev(config, corpus):
    model = Model(config) 
    assert os.path.exists(config.dev_model_path), 'Dev model does not exist : ' + config.dev_model_path
    # model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.load_state_dict(torch.load(config.dev_model_path)[model.model_name])
    model.cuda()
    dev_res_dir = os.path.join(config.dev_res_dir, config.dev_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(dev_res_dir):
        os.mkdir(dev_res_dir)
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, corpus, config.batch_size * 2 // config.world_size, 'dev', dev_res_dir + '/' + model.model_name + '.txt', config.dataset)
    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    return auc, mrr, ndcg5, ndcg10


def test(config, corpus):
    model = Model(config) 
    assert os.path.exists(config.test_model_path), 'Test model does not exist : ' + config.test_model_path
    # model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.load_state_dict(torch.load(config.test_model_path)[model.model_name])
    model.cuda()
    test_res_dir = os.path.join(config.test_res_dir, config.test_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(test_res_dir):
        os.mkdir(test_res_dir)
    print('test model path  : ' + config.test_model_path)
    print('test output file : ' + test_res_dir + '/' + model.model_name + '.txt')
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, corpus, config.batch_size, 'test', test_res_dir + '/' + model.model_name + '.txt', config.dataset)   # config.batch_size * 2
    
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    if config.mode == 'train':
        with open(config.result_dir + '/#' + str(config.run_index) + '-test', 'w') as result_f:
            result_f.write('#' + str(config.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
    elif config.mode == 'test':
        with open(config.test_output_file, 'w', encoding='utf-8') as f:
            f.write('#3' + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
        with open(config.result_dir + '/#' + str(config.run_index) + '-test', 'w') as result_f:
            result_f.write('#' + str(config.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
        

# main.py
if __name__ == '__main__':
    config = Config() # configuration
    data_corpus = Corpus(config) # load dataset corpus
    if config.mode == 'train':
        train(config, data_corpus)
        if config.news_encoder == 'LIME':
            model_name = f"{config.news_encoder}-{config.content_encoder}-{config.user_encoder}"
        else:
            model_name = f"{config.news_encoder}-{config.user_encoder}"
        config.test_model_path = os.path.join(config.best_model_dir, f'#{config.run_index}', model_name)
        test(config, data_corpus)
    elif config.mode == 'test':
        config.test_model_path = 'best_model/adressa/LIME-CROWN-CROWN/#32/LIME-CROWN-CROWN'
        config.test_output_file = 'test/res/adressa/LIME-CROWN-CROWN/best_model_adressa_LIME-CROWN-CROWN_#32_LIME-CROWN-CROWN/LIME-CROWN-CROWN.txt'
        config.run_index = 32
        test(config, data_corpus)
