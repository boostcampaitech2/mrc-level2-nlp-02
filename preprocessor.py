import re

class Preprocessor :
    def __init__(self, ) :
        pass

    def preprocess_train(self, dataset) :
        assert isinstance(dataset, dict)
        context = dataset['context']
        question = dataset['question']
        answer = dataset['answers']

        answer_start, answer_text = answer['answer_start'][0], answer['text'][0]
        context_prev = context[:answer_start]
        context_next = context[answer_start + len(answer_text):]

        context_prev = self.preprocess_context(context_prev)
        context_next = self.preprocess_context(context_next)

        answer_text = self.convert_foreign(answer_text)
        answer_pos = len(context_prev)

        context = context_prev + answer_text + context_next
        answer = {'answer_start' : [answer_pos], 'text' : [answer_text]}
        question = self.convert_foreign(question)

        dataset['context'] = context
        dataset['question'] = question
        dataset['answers'] = answer
        return dataset

    def preprocess_inf(self, dataset) :
        assert isinstance(dataset, dict)
        context = dataset['context']
        question = dataset['question']

        context = self.preprocess_context(context)
        question = self.convert_foreign(question)

        dataset['context'] = context
        dataset['question'] = question
        return dataset

    def preprocess_context(self, context) :
        context = self.remove_newline(context)
        context = self.remove_outrange(context)
        context = self.convert_foreign(context)
        return context

    def remove_newline(self, txt) :
        txt = txt.replace('\u3000', ' ')
        txt = txt.replace('\n', ' ')
        txt = txt.replace('\\n', ' ')
        txt = re.sub('\s+', ' ', txt)
        return txt
    
    def remove_outrange(self, txt) :
        txt = re.sub('[힣-\uffff]', ' ', txt)
        txt = re.sub('\s+', ' ', txt)
        return txt

    def convert_foreign(self, txt) :
        txt = re.sub('\\([\u4e00-\u9fff\u3040-\u31ff ]+\\)', '[CHN]', txt)
        return txt