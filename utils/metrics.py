import math
import numpy as np

def get_r_precision(answer, cand):
    set_answer = set(answer)
    r = len(set_answer&set(cand[:len(answer)])) / len(answer)
    return r

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

def get_metrics(answer,cand):
    r_precision = get_r_precision(answer,cand)
    ndcg = get_ndcg(answer,cand)
    rsc = get_rsc(answer,cand)
    
    return r_precision,ndcg,rsc

def single_eval(scores, seed, answer):
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
    rprecision, ndcg, rsc = get_metrics(answer,cand)
    return rprecision,ndcg,rsc