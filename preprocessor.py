import re

class Preprocessor :
    def __init__(self, ) :
        self.bracket_comp = re.compile(r'\([^)]*\)')
        self.chn_comp = re.compile('[一-鿕ぁ-ヿ]+')
        self.unicode_comp = re.compile('[' + chr(0) + '-' + chr(31) + chr(8191) + '-' + chr(12288) + chr(55204) + '-' + chr(63743) + ']')

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
        context = self.remove_special_unicode(context)
        context = self.convert_foreign(context)
        return context

    def remove_newline(self, txt) :
        """[summary] : remove '\n' code
        Args:
            txt ([str]): question and context
        Returns:
            [str]: text which '\n' characters is removed
        """
        txt = txt.replace('</br>', ' ')
        txt = txt.replace('\n*' , ' ')
        txt = txt.replace('\n#' , ' ')
        txt = txt.replace('\n' , ' ')
        txt = re.sub('\s+' , ' ', txt)
        return txt
    
    def remove_special_unicode(self, txt) :
        """[summary] : remove special unicode (e.g '\u3000' , '\xa0' ...)
        Args:
            txt ([str]): question and context
        Returns:
            [str]: Text which speical unicode is removed 
        """
        txt = self.unicode_comp.sub(' ', txt)
        txt = re.sub('\s+' , ' ', txt)
        return txt

    def convert_foreign(self, txt) :
        bracket_list = self.bracket_comp.finditer(txt)
        prev_brackets = []
        for bracket in bracket_list :
            start_idx, end_idx = bracket.start(), bracket.end()
            prev_brackets.append(txt[start_idx:end_idx])

        for bracket in prev_brackets :
            cur_bracket = self.chn_comp.sub('[CHN]', bracket)
            if bracket == cur_bracket :
                continue
            txt = txt.replace(bracket, cur_bracket)

        return txt