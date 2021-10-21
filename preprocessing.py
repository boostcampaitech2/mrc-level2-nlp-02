import re
from typing import Union, list
from datasets import DatasetDict
def preprocessing_data(data : Union[DatasetDict,list], do_train=False, do_eval=False) -> None:
    """
       wiki corpus, trainset, validationset의 context 혹은 text data 전처리를 위한 함수입니다.
       trainset 전처리 : do_train=True
       validationset 전처리 : do_eval=True 
    """
    #전처리 complier를 불러옵니다.
    p = preprocesing_compile()

    # train/eval data
    if type(data) == DatasetDict:
        if do_train==True:
            data['train']['context'] = list(map(lambda x : p.sub("",x),data['train']['context']))
            print("Train Dataset 전처리 완료")
        if do_eval==True:
            data['validation']['context'] = list(map(lambda x : p.sub("",x),data['validation']['context']))
            print("Validation Dataset 전처리 완료")

    # wiki corpus data
    elif type(data) == list:
        data = list(map(lambda x : p.sub("",x),data))
        print("Wiki Corpus 전처리 완료")

    # 불필요 데이터인 경우 error 반환
    else:
        raise TypeError("DatasetDict type이나 Dict type의 데이터를 입력해주세요")
    
    return data

def preprocesing_compile():
    return re.compile("(\\n)+|(\\\\n)+|(\xa0)|(\u3000)")