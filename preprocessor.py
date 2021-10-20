import re

class TextPreprocessor :
    def __init__(self, ) :
        ch_start_idx = int('4E00', 16)
        ch_end_idx = int('9FFF', 16)
        jp_start_idx = int('3040', 16)
        jp_end_idx = int('30FF', 16)
        self.ch_sub = re.compile('[' + chr(ch_start_idx) + '-' + chr(ch_end_idx) + ']+')
        self.jp_sub = re.compile('[' + chr(jp_start_idx) + '-' + chr(jp_end_idx) + ']+')

        self.unicode_comp1 = re.compile('[' + chr(0) + '-' + chr(31) + ']')
        self.unicode_comp2 = re.compile('[' + chr(8191) + '-' + chr(12288) + ']')
        self.unicode_comp3 = re.compile('[' + chr(55204) + '-' + chr(63743) + ']')

    def preprocess_cq(self, txt) :
        txt = self.remove_newline(txt)
        txt = self.remove_special_unicode(txt)
        txt = self.convert_foreign(txt)
        return txt

    def preprocess_a(self, txt) :
        txt = self.convert_foreign(txt)
        return txt

    def remove_newline(self, txt) :
        """[summary] : remove '\n' code
        Args:
            txt ([str]): question and context
        Returns:
            [str]: text which '\n' characters is removed
        """
        txt = txt.replace('\n*' , '')
        txt = txt.replace('\n#' , '')
        txt = txt.replace('\n' , '')
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
        txt = self.jp_sub.sub('[JPN]', txt)
        return txt