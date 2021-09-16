#!/usr/bin/env python3
"""
Function that calculates the n-gram BLEU score for a sentence
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    * references is a list of reference translations
        * each reference translation is a list of the words in the translation
    * sentence is a list containing the model proposed sentence
    * n is the size of the n-gram to use for evaluation
    * Returns: the n-gram BLEU score
    """
    senten_lenght = len(sentence)
    ref_len = []
    words = {}
    for transl in references:
        ref_len.append(len(transl))
        for word in transl:
            if word in sentence and word not in words.keys():
                    words[word] = 1
    s = len(sentence)
    total = sum(words.values())
    index = np.argmin([abs(len(i) - senten_lenght) for i in references])
    best_match = len(references[index])

    if sentence_length > best_match:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(best_match) / float(sentence_length))
    BLEU_score = BLEU * np.exp(np.log(total / sentence_length))

    return BLEU_score
