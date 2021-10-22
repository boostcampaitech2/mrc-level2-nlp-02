import re

class Preprocessor :
    def __init__(self, ) :
        ch_start_idx = int('4E00', 16)
        ch_end_idx = int('9FFF', 16)
        jp_start_idx = int('3040', 16)
        jp_end_idx = int('30FF', 16)

        self.ch_sub = re.compile(r'\([' + chr(ch_start_idx) + '-' + chr(ch_end_idx) + chr(jp_start_idx) + '-' + chr(jp_end_idx) + ' ]+\)')
        self.unicode_comp1 = re.compile('[' + chr(0) + '-' + chr(31) + ']')
        self.unicode_comp2 = re.compile('[' + chr(8191) + '-' + chr(12288) + ']')
        self.unicode_comp3 = re.compile('[' + chr(55204) + '-' + chr(63743) + ']')

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
        txt = txt.replace('</br>', '')
        txt = txt.replace(r'\n**' , '')
        txt = txt.replace(r'\n*' , '')
        txt = txt.replace(r'\n#' , '')
        txt = txt.replace(r'\n' , '')
        return txt
    
    def remove_special_unicode(self, txt) :
        """[summary] : remove special unicode (e.g '\u3000' , '\xa0' ...)
        Args:
            txt ([str]): question and context
        Returns:
            [str]: Text which speical unicode is removed 
        """
        txt = self.unicode_comp1.sub(' ', txt)
        txt = self.unicode_comp2.sub(' ', txt)
        txt = self.unicode_comp3.sub(' ', txt)
        txt = re.sub('\s+' , ' ', txt)
        return txt

    def convert_foreign(self, txt) :
        """[summary] : convert chinese word to [CHN] token, and japanese word to [JPN] token
        Args:
            txt ([str]): question, context and answer
        Returns:
            [str]: converted txt
        """
        txt = self.ch_sub.sub('[CHN]', txt)
        return txt


