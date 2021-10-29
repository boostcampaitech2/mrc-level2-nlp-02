import re
from datasets import DatasetDict
from transformers import BertTokenizerFast

class Preprocessor :
    pattern_dict={
                "1" : re.compile("(\\n)+|(\\\\n)+|(\\xa0)|(\\u3000)"),
                "2" : re.compile("[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥()?!∧≪≫『』\'<>〈〉:「」＜＞<>》《・\"-“”\s\.\‘’%,]"),
                "3" : re.compile(r'[\u0000-\u001f\u1fff-\u3000\ud7a4-\uf8ff\U000186a0-\U00030d40]'),
            }
    
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


class PreprocessorTokenizer :
    def __init__(self, tokenizer) :
        assert isinstance(tokenizer, BertTokenizerFast)
        self.tokenizer = tokenizer
        self.unk_id = tokenizer.convert_tokens_to_ids('[UNK]')
        unk_chars = []
        for i in range(10000) :
            if tokenizer.convert_tokens_to_ids(chr(i)) == self.unk_id :
                unk_chars.append(chr(i))
        self.unk_chars = re.compile('['+''.join(unk_chars) + ']')

    def preprocessing(self, data) :
        if isinstance(data , DatasetDict) :
            data = data.map(lambda x : self.reconstruct(dataset = x)) 
        elif isinstance(data, list) :
            data = list(map(lambda x : self.sen_preprocess(x), data))
        else :
            assert TypeError, "Wrong Data Type for Preprocessing"
        return data
   
    def reconstruct(self, dataset) :
        assert isinstance(dataset, dict)
        context = dataset['context']
        question = dataset['question']
        answer = dataset['answers'] 
        answer_start, answer_text = answer['answer_start'][0], answer['text'][0]
        context_prev = context[:answer_start]
        context_next = context[answer_start + len(answer_text):]

        context_prev = self.sen_preprocess(context_prev)
        context_next = self.sen_preprocess(context_next)

        question = self.sen_preprocess(question)
        answer_text = self.sen_preprocess(answer_text)

        answer_pos = len(context_prev)
        context = context_prev + answer_text + context_next
        answer = {'answer_start' : [answer_pos], 'text' : [answer_text]}

        dataset['context'] = context
        dataset['answers'] = answer
        return dataset

    def sen_preprocess(self, sen) :
        assert isinstance(sen, str)
        sen = re.sub(r'\n|\\n', '' , sen)
        sen = re.sub(r'[\U000186a0-\U00030d40]', '', sen)
        sen = self.unk_chars.sub(' ', sen)
        sen = re.sub('\s+', ' ', sen)
        return sen 
