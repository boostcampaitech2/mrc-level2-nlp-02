import re
from typing import Union, List
from datasets import DatasetDict
from konlpy.tag import Mecab
# def preprocessing_data(data : Union[DatasetDict,List], do_train=False, do_eval=False) -> None:

##정규표현식 패턴
pattern = "(\\n)+|(\\\\n)+|(\xa0)|(\u3000)"
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
        data = list(map(lambda x : re.sub(pattern,"",x), data))
        # data = list(map(lambda x : " ".join(m.morphs(x)), data))
        print(">>>>>>>>>>Wiki Corpus-context 전처리 완료")

    # 불필요 데이터인 경우 error 반환
    else:
        raise TypeError("DatasetDict type이나 Dict type의 데이터를 입력해주세요")
    return data

def preprocessing_context(datasets):
    preprocessed_context = re.sub(pattern,"",datasets['context'])
    # preprocessed_context = " ".join(m.morphs(datasets['context']))
    datasets['context'] = preprocessed_context
    return datasets