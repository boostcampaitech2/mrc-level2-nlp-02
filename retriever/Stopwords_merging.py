import os
import sys

dir = '/opt/ml/mrc-level2-nlp-02/retriever/Stopwords'
stop_files = os.listdir(dir)
stop_list = []

count = 0
for file in stop_files :
    if 'txt' not in file :
        continue
    print(file)
    with open(os.path.join(dir,file)) as f:
        for idx, item in enumerate(f) :
            if idx == 0 : # header
                continue
            else :
                count += 1
                item = item.rstrip()
                if len(item.split('\t')) > 1 :
                    stop_list.append(item[0])
                    break
                else :
                    stop_list.append(item)

unique_stop_list = sorted(list(set(stop_list)))
len(unique_stop_list)
with open(os.path.join(dir,'stopwords.txt'),'w') as f :
    for word in stop_list :
        f.write(word+'\n')