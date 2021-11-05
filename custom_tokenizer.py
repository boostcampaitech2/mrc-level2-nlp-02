from transformers import AutoTokenizer
from typing import Optional
from konlpy.tag import Mecab
from tqdm import tqdm
import os
import re

speical_tokens = {'additional_special_tokens': ['[SPLIT]', '[WHO]','[WHEN]','[WHERE]','[HOW]','[WHY]','[WHAT]']}


def load_pretrained_tokenizer(pretrained_model_name_or_path:str,
                              data_selected:Optional[str]= "",
                              datasets=None,
                              add_special_tokens_flag:bool=False,
                              use_fast:bool=True):
    """
        [UNK] token을 줄이기 위해서 train dataset을 기반으로 vocab을 추가한 새로운 토크나이저 제작 
    """
    
    #custom tokenizer name 지정
    tokenizer_name = "tokenizer/custom_" \
                        + ("c" if 'context'  in data_selected else "" ) \
                        + ("q" if 'question' in data_selected else "" ) \
                        + ("a" if 'answers'  in data_selected else "" )
                        
    if data_selected: #custom_flag=True인 경우 Custom_tokenizer 사용
        if not os.path.isdir(tokenizer_name):#동일한 custom 토크나이저가 존재한 경우 기존 tokenizer를 그대로 사용
            save_customized_tokenizer(datasets['train'], pretrained_model_name_or_path, data_selected, use_fast, tokenizer_name)
            print("make customized tokenizer!!!!!!!!!!!!")
        tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=use_fast)
    
    #special token이 필요한 경우 추가
    if add_special_tokens_flag:
        tokenizer.add_special_tokens(speical_tokens)
        tokenizer.save_pretrained(tokenizer_name)
    return tokenizer

def restore_sentence_by_tokens(tokenized_tokens):
    """
        Bert Tokenizer으로 tokenization한 결과를 sentence로 복원해주는 함수
    """
    sent = ""
    for token in tokenized_tokens:
        if "##" in token: # ##이 붙은 경우 떼고 문장을 복원합니다.
            sent +=re.sub("##","",token) 
        elif "[UNK]"==token or re.sub("[a-zA-Z0-9가-힣]","",token)!="": #  [UNK]/특수문자/다른 나라 언어인 경우 그대로 복원합니다.
            sent += token 
        else: #나머지 경우는 blank를 추가한 다음 복원합니다.
            sent +=" " + token
    return sent.strip() #마지막으로 양쪽의 빈칸을 제거한 다음 최종 문장을 출력합니다.


def get_added_token(trainset,
                    tokenizer,
                    data_selected:str):
    """
        Toeknizer에 새롭게 추가할 vocab list를 생성해주는 함수
        1. Mecab으로 sentence 분리
        2. Bert tokenizer로 형태소를 subword 단위로 분리
        3. 원래 문장과 restore_word_by_tokens으로 복원한 문장에서 없는 문자인 경우 bi-gram subword 방식으로 만들어서 vocab list에 추가
        e.g.) "홍길동"이 UNK인 경우 홍, ##길, ##동
        
    """
    
    mecab = Mecab()

    chn_char_stt = int("4E00",16) #start 한자 유니코드
    chn_char_end = int("9FFF",16) #end 한자 유니코드
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
        
        # mecab 형태소 분류기로 answer 분리합니다.
        tokens_mecab = mecab.morphs(text)
        tokens_bert = []

        # 분리된 answer를 Berttokenizer로 subword 단위로 분리합니다.
        for token_mecab in tokens_mecab:
            tokens_bert.extend(tokenizer.tokenize(token_mecab))
        
        # # UNK가 있는 sequence가 있는 경우에만 진행합니다.
        if "[UNK]" in tokens_bert:
            unk_sent.append(text)
            token_flag = False #Flag=False면 그대로, True 이후 음절은 ## 추가합니다.
            resotred_answer = restore_sentence_by_tokens(tokens_bert) #Mecab-BertTokenzer를 통과한 subword를 문장으로 복원, 단 UNK 존재합니다.
            for s in restore_sentence_by_tokens(tokens_mecab) : #Mecab으로 분류된 형태소 토큰을 문장으로 복원합니다.
                if s in resotred_answer: #음절 단위로 비교, 문장 내에 존재하면 pass
                    token_flag = False
                    continue
                
                # 한자 인 경우
                if chn_char_stt <= ord(s) <= chn_char_end: # UNK인 subword가 한자인 경우 추가합니다.
                    added_token_set.add(s)
                # 한자가 아닌 경우
                elif token_flag == False:
                    added_token_set.add(s)
                    token_flag = True
                else:
                    added_token_set.add("##"+s)
    return list(added_token_set)

def save_customized_tokenizer(trainset,
                              pretrained_model_name_or_path:str,
                              data_selected:str,
                              use_fast:bool,
                              tokenizer_name:str):
    """
        새로 생성한 customized된 bert tokenizer를 저장해주는 함수
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=use_fast)
    
    added_token_list = get_added_token(trainset, tokenizer, data_selected)
    tokenizer.add_tokens(added_token_list)

    #tokenizer를 저장합니다.
    tokenizer.save_pretrained(tokenizer_name)