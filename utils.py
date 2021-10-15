import json
from tokenizers import Tokenizer

def is_not_roberta(tokenizer: Tokenizer) -> bool:
    """해당 tokenizer가 roberta 토크나이저인지 판별한다.

    Args:uti
        tokenizer (Tokenizer): 검사하고자 하는 tokenizer

    Returns:
        bool: roberta tokenizer면 False반환 아니면 True 반환
    """
    
    try:
        path = tokenizer.name_or_path
        with open(path) as f:
            tokenizer_info = str(json.load(f))
        if 'roberta' in tokenizer_info:
            return False
        else:
            return True
    except:
        if 'roberta' in tokenizer.name_or_path:
            return False
        else:
            return True