from collections import defaultdict
from huggingface_utils import MODELS
from sklearn.utils import shuffle
from transformers import *
#from ust import train_model
import pandas as pd
import numpy as np
from ust import train_model
import random
import logging
import os
logger = logging.getLogger('UST')
logging.basicConfig(level = logging.INFO)
class UST_trainer:
  def __init__(self,model_dir,train_file_path,train_t_col,train_f_col,test_file_path,test_t_col,test_f_col, transfer_file_path, transfer_f_col,seq_len, val_split=.2, sup_batch_size=4, unsup_batch_size=32, 
               sample_size=16384, unsup_size=4096, sample_scheme="easy_bald_class_conf", sup_labels=60, T=30, alpha=.1, valid_split=.5, sup_epochs=70,
               unsup_epochs=25, N_base=5, pt_teacher="TFBertModel", pt_teacher_checkpoint="bert-base-uncased", do_pairwise=False, hidden_dropout=.2, 
               attention_probs_dropout_prob=.2, dense_dropout=.5,GLOBAL_SEED=None):
    
    self.train_config={'train_file_path':train_file_path,'train_t_col':train_t_col,'train_f_col':train_f_col}
    self.test_config={'test_file_path':test_file_path,'test_t_col':test_t_col,'test_f_col':test_f_col}
    self.transfer_config={'transfer_file_path':transfer_file_path, 'transfer_f_col':transfer_f_col}
    self.config={'model_dir':model_dir,'seq_len':seq_len, 'val_split':val_split, 'sup_batch_size':sup_batch_size, 'unsup_batch_size':unsup_batch_size, 
               'sample_size':sample_size, 'unsup_size':unsup_size, 'sample_scheme':sample_scheme, 'sup_labels':sup_labels, 'T':T, 'alpha':alpha, 'valid_split':valid_split, 'sup_epochs':sup_epochs,
               'unsup_epochs':unsup_epochs, 'N_base':N_base, 'pt_teacher':pt_teacher, 'pt_teacher_checkpoint':pt_teacher_checkpoint, 'do_pairwise':False, 'hidden_dropout':hidden_dropout, 
               'attention_probs_dropout_prob':attention_probs_dropout_prob, 'dense_dropout':dense_dropout,'GLOBAL_SEED':GLOBAL_SEED}
  
  
  def get_model_data(self):
    for index, model in enumerate(MODELS):
      if model[0].__name__== self.config['pt_teacher']:
        tf_model, Tokenizer, config= MODELS[index]
    tokenizer= Tokenizer.from_pretrained(self.config['pt_teacher_checkpoint'])
    return tf_model, tokenizer, config

  
  
  def generate_sequence_data(self,seq_len,file_path, tokenizer,target_col=None,feature_col=None,unlabeled=False, do_pairwise=False):
      X1 = []
      X2 = []
      y = []
      label_count = defaultdict(int)
      if feature_col==None and target_col==None:
        df=pd.read_csv(file_path,header=None)
      else:
        df=pd.read_csv(file_path)
        if unlabeled:
          df=df[feature_col]
        else:
          df=df[[feature_col, target_col]]
      for i in range(len(df)):
        line=df.iloc[i]
        if len(line) == 0:
          continue
        X1.append(line[0])
        if do_pairwise:
          X2.append(line[1])
        if not unlabeled:
            if do_pairwise:
              label = int(line[2])
            else:
              label = int(line[1])
            y.append(label)
            label_count[label] += 1
        else:
            y.append(-1)
      if do_pairwise:
        X =  tokenizer(X1, X2, padding=True, truncation=True, max_length = seq_len)
      else:
        X =  tokenizer(X1, padding=True, truncation=True, max_length = seq_len)

      for key in label_count.keys():
          logger.info ("Count of instances with label {} is {}".format(key, label_count[key]))

      if "token_type_ids" not in X:
          token_type_ids = np.zeros((len(X["input_ids"]), seq_len))
      else:
          token_type_ids = np.array(X["token_type_ids"])
      return {"input_ids": np.array(X["input_ids"]), "token_type_ids": token_type_ids, "attention_mask": np.array(X["attention_mask"])}, np.array(y)

  def train(self):
    #getting the model and the tokenizer
    model,tokenizer, config= self.get_model_data()
    # converting train, test and unabelled into respective datasets
    X_train_all, y_train_all = self.generate_sequence_data(self.config['seq_len'], self.train_config['train_file_path'], tokenizer, feature_col=self.train_config['train_f_col'],target_col=self.train_config['train_t_col'], unlabeled=False, do_pairwise=self.config['do_pairwise'])
    X_test, y_test = self.generate_sequence_data(self.config['seq_len'], self.test_config['test_file_path'], tokenizer, feature_col=self.test_config['test_f_col'],target_col=self.test_config['test_t_col'], do_pairwise=self.config['do_pairwise'])
    X_unlabeled, _ = self.generate_sequence_data(self.config['seq_len'],self.transfer_config['transfer_file_path'],tokenizer, feature_col=self.transfer_config['transfer_f_col'], unlabeled=True, do_pairwise=self.config['do_pairwise'])
    labels = set(y_train_all)
    if 0 not in labels:
      y_train_all -= 1
      y_test -= 1 	
    labels = set(y_train_all)	
    #if sup_labels < 0, then use all training labels in train file for learning
    sup_labels=self.config["sup_labels"]
    if sup_labels < 0:
      X_train = X_train_all
      y_train = y_train_all
    else:
      X_input_ids, X_token_type_ids, X_attention_mask, y_train = [], [], [], []
      for i in labels:
        #get sup_labels from each class
        indx = np.where(y_train_all==i)[0]
        if not self.config["GLOBAL_SEED"]:
          random.shuffle(indx)
        else:
          random.Random(self.config["GLOBAL_SEED"]).shuffle(indx)
        indx = indx[:sup_labels]
        X_input_ids.extend(X_train_all["input_ids"][indx])
        X_token_type_ids.extend(X_train_all["token_type_ids"][indx])
        X_attention_mask.extend(X_train_all["attention_mask"][indx])
        y_train.extend(np.full(sup_labels, i))
        X_input_ids, X_token_type_ids, X_attention_mask, y_train = shuffle(X_input_ids, X_token_type_ids, X_attention_mask, y_train, random_state=self.config['GLOBAL_SEED'])

      X_train = {"input_ids": np.array(X_input_ids), "token_type_ids": np.array(X_token_type_ids), "attention_mask": np.array(X_attention_mask)}
      y_train = np.array(y_train)
    train_model(self.config['seq_len'], X_train, y_train, X_test, y_test, X_unlabeled, self.config['model_dir'], tokenizer, sup_batch_size=self.config['sup_batch_size'], unsup_batch_size=self.config['unsup_batch_size'], 
                  unsup_size=self.config['unsup_size'], sample_size=self.config['sample_size'], TFModel=model, Config=config, pt_teacher_checkpoint=self.config['pt_teacher_checkpoint'], sample_scheme=self.config['sample_scheme'], 
                  T=self.config['T'], alpha=self.config['alpha'], valid_split=self.config['valid_split'], sup_epochs=self.config['sup_epochs'], unsup_epochs=self.config['unsup_epochs'], N_base=self.config['N_base'], 
                  dense_dropout=self.config['dense_dropout'], attention_probs_dropout_prob=self.config['attention_probs_dropout_prob'], hidden_dropout_prob=self.config['hidden_dropout'])