from transformers import AutoTokenizer
from konlpy.tag import Mecab
from tqdm import tqdm
import os
import re

speical_tokens = {'additional_special_tokens': ['[CHN]']}
Q_special_tokens = {'additional_special_tokens': ['[WHO]','[WHEN]','[WHERE]','[HOW]','[WHY]','[WHAT]']}

def load_pretrained_tokenizer(pretrained_model_name_or_path:str,
                              tokenizer_name:str,
                              custom_flag:bool=False,
                              data_selected:str= None,
                              datasets=False,
                              add_special_tokens_flag=False,
                              add_special_tokens_query_flag=False,
                              use_fast=True):
    
    if custom_flag: #custom_flag=True인 경우 Custom_tokenizer 사용
        if not os.path.isdir(tokenizer_name):
            save_customized_tokenizer(datasets['train'], pretrained_model_name_or_path, data_selected,
                                      use_fast,tokenizer_name,add_special_tokens_flag,add_special_tokens_query_flag)
            print("make customized tokenizer!!!!!!!!!!!!")
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast)
    else:
        
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=use_fast)
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        if add_special_tokens_flag == True :
            add_special_tokens(tokenizer)
        
        if add_special_tokens_query_flag == True:
            add_special_tokens_query(tokenizer)
        return tokenizer


def restore_word_by_tokens(tokenized_tokens):
    word = ""
    for token in tokenized_tokens:
        if "##" in token: # ##이 붙은 경우 ##을 떼고 입력
            word +=re.sub("##","",token) 
        elif "[UNK]"==token or re.sub("[a-zA-Z0-9가-힣]","",token)!="": #특수문자나 다른 나라 언어인 경우 그대로 입력
            word += token 
        else: #나머지 경우는 앞의 한 칸을 띄고 입력
            word +=" " + token
    return word.strip() #양쪽 빈칸 제거


def get_added_token(trainset, tokenizer, data_selected):
    mecab = Mecab()

    chn_char_stt = int("4E00",16) #한자 유니코드 
    chn_char_end = int("9FFF",16) #한자 유니코드
    added_token_set = set()
    unk_sent = []
    
    data_list = []
    data_selected = data_selected.split("_")
    data_list += trainset['context'] if "context" in data_selected else ""
    data_list += trainset['answers'] if "answers" in data_selected else ""
    data_list += trainset['question'] if "question" in data_selected else ""
    print("UNK 토큰 리스트 생성 중...")
    for sent in tqdm(data_list):
        try:
            text = sent['text'][0] #input : answers
        except:
            text = sent
        
        # mecab 형태소 분류기로 answer 분리
        tokens_mecab = mecab.morphs(text)
        tokens_bert = []

        # 분리된 answer를 Berttokenizer로 subword 단위로 분리
        for token_mecab in tokens_mecab:
            tokens_bert.extend(tokenizer.tokenize(token_mecab))
        
        # # UNK가 있는 sequence가 있는 경우에만 진행
        if "[UNK]" in tokens_bert:
            unk_sent.append(text)
            token_flag = False #Flag=False면 그대로, True 이후 음절은 ## 추가
            resotred_answer = restore_word_by_tokens(tokens_bert) #Mecab-BertTokenzer를 통과한 subword를 문장으로 복원, 단 UNK 존재함
            for s in restore_word_by_tokens(tokens_mecab) : #Mecab으로 분류된 형태소 토큰을 문장으로 복원
                if s in resotred_answer: #음절 단위로 비교, 문장 내에 존재하면 pass
                    token_flag = False
                    continue
                
                # 한자 인 경우
                if chn_char_stt <= ord(s) <= chn_char_end: # UNK인 subword가 한자인 경우 추가
                    added_token_set.add(s)
                # 한자가 아닌 경우
                elif token_flag == False:
                    added_token_set.add(s)
                    token_flag = True
                else:
                    added_token_set.add("##"+s)
    return list(added_token_set)


def add_special_tokens(tokenizer):
    special_tokens_dict = speical_tokens
    return tokenizer.add_special_tokens(special_tokens_dict)  

def add_special_tokens_query(tokenizer):
    special_tokens_dict = Q_special_tokens
    return tokenizer.add_special_tokens(special_tokens_dict)

def save_customized_tokenizer(trainset, pretrained_model_name_or_path, data_selected,
                              use_fast, tokenizer_name,add_special_tokens_flag
                             ,add_special_tokens_query_flag):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=use_fast)
    
    added_token_list = get_added_token(trainset, tokenizer, data_selected)
    tokenizer.add_tokens(added_token_list)
    
    if add_special_tokens_flag==True:
        tokenizer = add_special_tokens(tokenizer)
    
    if add_special_tokens_query_flag==True:
        tokenizer = add_special_tokens_query(tokenizer)

    #tokenizer 저장
    tokenizer.save_pretrained(tokenizer_name)
