
import copy
import numpy as np
from datasets import Dataset

class SpanAugmentation :
    def __init__(self, n=4, p=0.8, max_len=24, min_len=6) :
        self.n = n
        self.p = p
        self.max_len = max_len
        self.min_len = min_len

    def __call__(self, dataset) :
        assert isinstance(dataset, Dataset)
        data_list = [data for data in dataset]
        data_size = len(data_list)

        data_augmented = []
        data_ans_text = set()
        rand_prob = np.random.rand(data_size)
        for i, data in enumerate(data_list) :
            if rand_prob[i] > self.p :
                continue
            for j in range(self.n) :
                direction = np.random.randint(3)
                if direction == 0 : # left
                    data_aug = self.left_augmentation(data,j)
                elif direction == 1 : # right
                    data_aug = self.right_augmentation(data,j)
                else : # mid
                    data_aug = self.mid_augmentation(data,j)

                ans_text = data_aug['answers']['text'][0] 
                if ans_text not in data_ans_text :
                    data_augmented.append(data_aug)
                    data_ans_text.add(ans_text)

        data_augmented = self.convert_to_dict(data_augmented)
        data_augmented = Dataset.from_dict(data_augmented)
        return data_augmented

    def convert_to_dict(self, data_list) :
        data_dict = {}
        for key in data_list[0].keys() :
            val_list = [data[key] for data in data_list]
            data_dict[key] = val_list
        return data_dict
    
    def left_augmentation(self, data, j) :
        context = data['context']
        answer = data['answers']
        answer_pos, answer_text = answer['answer_start'][0], answer['text'][0]

        answer_pos_left = max(0,answer_pos - np.random.randint(self.min_len, self.max_len))
        while context[answer_pos_left] != ' ' and answer_pos_left >= 0:
            answer_pos_left -= 1
        answer_pos_left += 1
        answer_text_left = context[answer_pos_left:answer_pos+len(answer_text)] 

        data_left = copy.deepcopy(data)
        data_left['id'] = data['id'] + '_0_' + str(j)
        data_left['answers'] = {'answer_start' : [answer_pos_left], 'text' : [answer_text_left]}
        return data_left

    def right_augmentation(self, data, j) :
        context = data['context']
        answer = data['answers']
        answer_pos, answer_text = answer['answer_start'][0], answer['text'][0]

        answer_pos_right = min(len(context)-1,answer_pos + len(answer_text) + np.random.randint(self.min_len, self.max_len))
        while context[answer_pos_right] != ' ' and answer_pos_right < len(context)-1:
            answer_pos_right += 1
        answer_text_right = context[answer_pos:answer_pos_right+1] 

        data_right = copy.deepcopy(data)
        data_right['id'] = data['id'] + '_1_' + str(j)
        data_right['answers'] = {'answer_start' : [answer_pos], 'text' : [answer_text_right]}
        return data_right

    def mid_augmentation(self, data, j) :   
        context = data['context']
        answer = data['answers']
        answer_pos = answer['answer_start'][0]

        answer_pos_left = max(0,answer_pos - np.random.randint(self.min_len/2, self.max_len/2))
        while context[answer_pos_left] != ' ' and answer_pos_left >= 0 :
            answer_pos_left -= 1
        answer_pos_left += 1

        answer_pos_right = min(len(context)-1,answer_pos + np.random.randint(self.min_len/2, self.max_len/2))
        while context[answer_pos_right] != ' ' and answer_pos_right < len(context)-1:
            answer_pos_right += 1

        answer_text = context[answer_pos_left:answer_pos_right+1] 
        data_mid = copy.deepcopy(data)
        data_mid['id'] = data['id'] + '_2_' + str(j)
        data_mid['answers'] = {'answer_start' : [answer_pos], 'text' : [answer_text]}
        return data_mid
