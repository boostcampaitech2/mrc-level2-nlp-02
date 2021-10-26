import re
from typing import Union, List
from datasets import DatasetDict
from konlpy.tag import Mecab

##정규표현식 패턴
pattern = "(\\n)+|(\\\\n)+|(\\xa0)|(\\u3000)"
# pattern = ""
m = Mecab()

def preprocessing_data(data):
    """
       wiki corpus, trainset, validationset의 context 혹은 text data 전처리를 위한 함수입니다.
       trainset 전처리 : do_train=True
       validationset 전처리 : do_eval=True 
    """

    # train/eval data
    if type(data) == DatasetDict:
        data = data.map(preprocessing_context)
        print(">>>>>>>>>>Dataset-context 전처리 완료")

    # wiki corpus data
    elif type(data) == list:
        data = list(map(lambda x : re.sub(pattern," ",x), data))
        print(">>>>>>>>>>Wiki Corpus-context 전처리 완료")

    # 불필요 데이터인 경우 error 반환
    else:
        raise TypeError("DatasetDict type이나 Dict type의 데이터를 입력해주세요")
    return data

def preprocessing_context(datasets):
    #context
    context = datasets['context']
    
    #answers
    answer = datasets['answers']
    answer_start, answer_text, answer_len = answer['answer_start'][0], answer['text'][0], len(answer['text'][0])
    
    #context without text in answer
    context_before_answer = context[:answer_start]
    context_after_answer = context[answer_start+answer_len:]

    context_before_answer = re.sub(pattern," ", context_before_answer)
    context_after_answer = re.sub(pattern," ", context_after_answer)

    preprocessed_context = context_before_answer + answer_text + context_after_answer
    # preprocessed_context = " ".join(m.morphs(context_before_answer)) + answer_text + " ".join(m.morphs(context_after_answer))

    #new answer start position
    new_answer_start = len(context_before_answer)

    answer = {"answer_start" : [new_answer_start], "text" : [answer_text]} 
    datasets['context'] = preprocessed_context
    datasets['answers'] = answer
    return datasets