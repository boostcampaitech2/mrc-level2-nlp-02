import re
from datasets import DatasetDict
import pandas as pd

class Preprocessor :
    pattern_dict={
                "1" : re.compile("(\\n)+|(\\\\n)+|(\\xa0)|(\\u3000)"),
                "2" : re.compile("(\\\\n)+|(\\n)+|[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥()?!∧≪≫『』\'<>〈〉:「」＜＞<>》《・\"-“”\s\.\‘’%,]"),
                "3" : re.compile('['+chr(0)+'-'+chr(31)+chr(8191)+'-'+chr(12288)+chr(55204)+'-'+chr(63743)+']')} # e.g \u3000 \u200d \u210e ...
    

    @classmethod
    def preprocessing(self, data, pt_num):
        # dataset
        if type(data) == DatasetDict:
            # data = data.map(self.reconstruct(pt_num=pt_num))        
            data = data.map(lambda x : self.reconstruct(self, dataset = x, pt_num=pt_num))
        
        # wiki corpus data
        elif type(data) == list:
            pd_data = pd.DataFrame({"contexts" : data})
            for num in "1":
                preprocessing = lambda x : self.pattern_dict[num].sub(" ", x)
                pd_data["contexts"] = pd_data.contexts.map(preprocessing)
            data = pd_data.drop_duplicates("contexts").contexts.to_list()
                # new_data += [preprocessing(d) for d in data]
                # data = list(map(lambda x : self.pattern_dict[num].sub(" ",x),data))
        return data
        
    def reconstruct(self, dataset, pt_num) :
        assert isinstance(dataset, dict)
        context = dataset['context']
        answer = dataset['answers'] 
        answer_start, answer_text = answer['answer_start'][0], answer['text'][0]
        context_prev = context[:answer_start]
        context_next = context[answer_start + len(answer_text):]

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
        return context

