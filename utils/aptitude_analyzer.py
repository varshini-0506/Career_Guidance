import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer #convert raw text into numerical features

def score_aptitude(answer_texts):
    keywords_logic = ['if', 'loop', 'function', 'complexity']
    keywords_comm = ['presentation', 'clarity', 'audience']

    logic_score = sum(any(k in a.lower() for k in keywords_logic) for a in answer_texts) / len(answer_texts)
    comm_score = sum(any(k in a.lower() for k in keywords_comm) for a in answer_texts) / len(answer_texts)

    return {
        "logic": round(logic_score, 2),
        "communication": round(comm_score, 2)
    }


