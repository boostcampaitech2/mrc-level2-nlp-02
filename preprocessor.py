import re
import pandas as pd
from datasets import DatasetDict
from transformers import BertTokenizerFast

class Preprocessor :
    pattern_dict={
                "1" : re.compile("(\\n)+|(\\\\n)+|(\\xa0)|(\\u3000)"),
                "2" : re.compile("(\\\\n)+|(\\n)+|[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥()?!∧≪≫『』\'<>〈〉:「」＜＞<>》《・\"-“”\s\.\‘’%,]"),
                "3" : re.compile(r'[\u0000-\u001f\u1fff-\u3000\ud7a4-\uf8ff\U000186a0-\U00030d40]'),
            }
    
    @classmethod
    def preprocessing(self, data, pt_num):
        # dataset
        if type(data) == DatasetDict:
            # data = data.map(self.reconstruct(pt_num=pt_num))        
            data = data.map(lambda x : self.reconstruct(self, dataset = x, pt_num=pt_num))
        
        # wiki corpus data 변경 필요
        elif type(data) == list:
            pd_data = pd.DataFrame({"contexts" : data})
            for num in pt_num:
                preprocessing = lambda x : self.pattern_dict[num].sub(" ", x)
                pd_data["contexts"] = pd_data.contexts.map(preprocessing)

            pd_data["contexts"] = pd_data.contexts.map(lambda x : re.sub('\s+', ' ', x))
            data = pd_data.drop_duplicates("contexts").contexts.to_list()
        return data

    def reconstruct(self, dataset, pt_num) :
        assert isinstance(dataset, dict)
        context = dataset['context']
        answer = dataset['answers'] 
        answer_start, answer_text = answer['answer_start'][0], answer['text'][0]
        context_prev = context[:answer_start]
        context_next = context[answer_start + len(answer_text):]

        answer_text = self.sen_preprocess(self, context = answer_text, pt_num=pt_num)
        context_prev = self.sen_preprocess(self, context=context_prev, pt_num=pt_num)
        context_next = self.sen_preprocess(self, context=context_next, pt_num=pt_num)

        answer_pos = len(context_prev)
        context = context_prev + answer_text + context_next
        answer = {'answer_start' : [answer_pos], 'text' : [answer_text]}

        dataset['context'] = context
        dataset['answers'] = answer
        return dataset

    def sen_preprocess(self, context, pt_num) :
        for num in pt_num:
            context = self.pattern_dict[num].sub(" ",context) 
            context = re.sub('\s+' , ' ', context)   
        return context

