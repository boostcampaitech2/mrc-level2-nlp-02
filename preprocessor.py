import re
from datasets import DatasetDict

class Preprocessor :
    pattern_dict={
                "1" : re.compile("(\\n)+|(\\\\n)+|(\\xa0)|(\\u3000)"),
                "2" : re.compile("[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥()?!∧≪≫『』\'<>〈〉:「」＜＞<>》《・\"-“”\s\.\‘’%,]"),
                "3" : re.compile('['+chr(0)+'-'+chr(31)+chr(8191)+'-'+chr(12288)+chr(55204)+'-'+chr(63743)+']')}
    
    chn_compiler = re.compile('[ぁ-ゔァ-ヴー々〆〤一-龥]+')
    bracket_compiler = re.compile(r'\(([^)]*)') 

    @classmethod
    def preprocessing(self, data, pt_num, chn_flag):
        # dataset
        if type(data) == DatasetDict:
            # data = data.map(self.reconstruct(pt_num=pt_num))        
            data = data.map(lambda x : self.reconstruct(self, dataset = x, pt_num=pt_num, chn_flag=chn_flag))
        
        # wiki corpus data 변경 필요
        elif type(data) == list:
            for num in pt_num:
                data = list(map(lambda x : self.pattern_dict[num].sub(" ", x), data))
            if chn_flag == True :
                data = list(map(lambda x : self.convert_chn(self, context=x), data))

        return data
        
    def reconstruct(self, dataset, pt_num, chn_flag) :
        assert isinstance(dataset, dict)
        context = dataset['context']
        answer = dataset['answers'] 
        answer_start, answer_text = answer['answer_start'][0], answer['text'][0]
        context_prev = context[:answer_start]
        context_next = context[answer_start + len(answer_text):]

        context_prev = self.sen_preprocess(self, context=context_prev, pt_num=pt_num, chn_flag=chn_flag)
        context_next = self.sen_preprocess(self, context=context_next, pt_num=pt_num, chn_flag=chn_flag)

        answer_pos = len(context_prev)

        if chn_flag == True :
            question = dataset['question']
            question = self.convert_chn(self, question)
            answer_text = self.convert_chn(self, answer_text)
            dataset['question'] = question

        context = context_prev + answer_text + context_next
        answer = {'answer_start' : [answer_pos], 'text' : [answer_text]}

        dataset['context'] = context
        dataset['answers'] = answer
        return dataset

    def sen_preprocess(self, context, pt_num, chn_flag) :
        for num in pt_num:
            context = self.pattern_dict[num].sub(" ",context)
        if chn_flag == True :
            context = self.convert_chn(self, context)        
        return context

    def convert_chn(self, context) :
        prev_words = []
        cur_words = []

        bracket_list = self.bracket_compiler.finditer(context)

        for bracket in bracket_list :
            start_idx, end_idx = bracket.start(), bracket.end()

            prev_bracket = context[start_idx:end_idx+1]
            prev_words.append(prev_bracket)
            cur_bracket = self.chn_compiler.sub('[CHN]', prev_bracket)
            cur_words.append(cur_bracket)
        
        bracket_size = len(prev_words)
        for i in range(bracket_size) :
            context = context.replace(prev_words[i], cur_words[i])
        return context