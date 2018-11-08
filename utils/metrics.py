import math
import numpy as np


def get_class(class_divpnt, idx):
    for c in class_divpnt:
        if idx <= c:
            return class_divpnt.index(c)
    return len(class_divpnt)


def get_class_dist(cls_list, num_cls):
    cls_dist = [1e-9] * num_cls
    for i in cls_list:
        if i is not -1:
            cls_dist[i]+=1
    return cls_dist


def get_r_precision(answer, cand, answer_cls, class_divpnt):
    num_cls = len(class_divpnt) + 1
    hr_by_cls = [0] * num_cls
    cls_dist = get_class_dist(answer_cls, num_cls)

    set_answer = set(answer)
<<<<<<< HEAD
    r = len(set_answer&set(cand[:len(answer)])) / len(answer)
    return r
=======
    cand = cand[:len(set_answer)]
    cand_cls = []
    for c in cand:
        cls = get_class(class_divpnt, c)
        cand_cls.append(cls)

    _cand_cls_dist = get_class_dist(cand_cls, num_cls)
    s = sum(_cand_cls_dist)
    cand_cls_dist = [i/s for i in _cand_cls_dist]


    s = set_answer & set(cand)
    r = len(s) / len(set_answer)
    lst = list(s)
    for id in lst:
        cls = answer_cls[answer.index(id)]
        hr_by_cls[cls]+=1
    for i in range(num_cls):
        hr_by_cls[i] = hr_by_cls[i] / cls_dist[i]

    return r, hr_by_cls, cand_cls_dist
>>>>>>> e699f7b47d3f20b6e90a780eb2af4224ea75800b

def get_ndcg(answer, cand):
    cand_len = len(cand) 
    idcg=1
    idcg_idx=2
    dcg=0
    if cand[0] in answer:  dcg=1
    
    for i in range(1,cand_len):
        if cand[i] in answer: 
            dcg += (1/math.log(i+1,2))
            idcg += (1/math.log(idcg_idx,2))
            idcg_idx+=1
    
    return dcg/idcg

def get_rsc(answer, cand):
    cand_len = len(cand)
    for i in range(cand_len):
        if cand[i] in answer:
            return i//10
    return 51

def get_metrics(answer,cand, answer_cls, num_cls):
    r_precision, hr_by_cls, cand_cls_dist = get_r_precision(answer,cand, answer_cls, num_cls)
    # ndcg = get_ndcg(answer,cand)
    # rsc = get_rsc(answer,cand)
    
    return r_precision, hr_by_cls, cand_cls_dist

def single_eval(scores, seed, answer, answer_cls, num_cls):
    cand = np.argsort(-1*scores)
    cand = cand.tolist()
    #print("sort:",np.sort(-1*scores)[:10])
    #print("cand:",cand[:10])
    for i in seed:
        try:
            cand.remove(i)
        except:
            pass
    cand = cand[:500]
    rprecision, hr_by_cls, cand_cls_dist = get_metrics(answer,cand, answer_cls, num_cls)
    return rprecision, hr_by_cls, cand_cls_dist