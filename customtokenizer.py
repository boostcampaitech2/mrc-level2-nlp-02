from transformers import AutoTokenizer, BertTokenizer
from collections import deque
from konlpy.tag import Mecab
from tqdm import tqdm
import os
import re

def load_pretrained_tokenizer(pretrained_model_name_or_path:str,
                              custom_flag:bool=False,
                              datasets=False,
                              use_fast=False):
    
    if custom_flag: #custom_flag=True인 경우 Custom_tokenizer 사용
        if not os.path.isdir(custom_tokenizer_dict[pretrained_model_name_or_path]):
            save_customized_tokenizer(datasets['train'], pretrained_model_name_or_path)
            print("make customized tokenizer!!!!!!!!!!!!")
        return AutoTokenizer.from_pretrained(custom_tokenizer_dict[pretrained_model_name_or_path], use_fast=use_fast)
    else:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=use_fast)
    # else:
    #     return AutoTokenizer.from_pretrained(pretrained_model_name_or_path,use_fast=use_fast)
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.


custom_tokenizer_dict = {
    # "klue/roberta-large" : "./robertatokenizer_customized"
    "klue/roberta-large" : "./roberta_customized_context_based"
}


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


def get_added_token(trainset, tokenizer, mecab):
    #기본 토크나이저 및 mecab
    # tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
    # m = Mecab()

    chn_char_stt = int("4E00",16) #한자 유니코드 
    chn_char_end = int("9FFF",16) #한자 유니코드
    added_token_set = set()
    unk_sent = []
    print("추가할 UNK 토큰 리스트 생성 중...")
    for sents in [
        trainset['context']
        # trainset['answers'],
        # trainset['question'],
        ]:
        for sent in tqdm(sents):
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
    print("number of tokens added : ",len(added_token_set))
    return list(added_token_set)


def save_customized_tokenizer(trainset, pretrained_model_name_or_path):
    #기본 토크나이저 및 mecab 호출
    mecab = Mecab()
    # tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    #for BertTokenizer
    # #""[unusedxxx] token 리스트 만들기
    # unused_vocab_list = deque()
    # for v,i in tokenizer.vocab.items():
    #     if "unused" in v:
    #         unused_vocab_list.append((v,i))
    
    # # tokenizer vocab의 unused token을 새로운 token으로 변경
    # for added_token in get_added_token(trainset, tokenizer, mecab):
    #     if added_token not in tokenizer.vocab.keys():
    #         del_key, num = unused_vocab_list.popleft()
    #         del tokenizer.vocab[del_key]
    #         tokenizer.vocab[added_token] = num
    #for AutoTokenizer
    added_token_list = get_added_token(trainset, tokenizer, mecab)
    tokenizer.add_tokens(added_token_list)

    #tokenizer 저장
    tokenizer.save_pretrained(custom_tokenizer_dict[pretrained_model_name_or_path])
