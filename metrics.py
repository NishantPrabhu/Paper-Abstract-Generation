
""" 
Some metrics to monitor model progress.

Planning to add:
    1. Average generated text length
    2. BLEU score @ 1, 2, 3, 4
    3. Perplexity [Difficult so might skip]
"""

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


def bleu_score(output_texts, target_texts):
    target_words, output_words = [], []
    assert len(output_texts) == len(target_texts), \
        f"[BLEU] {len(output_texts)} sentences in output, {len(target_texts)} sentences in target"

    for i in range(len(output_texts)):
        output_words.append(output_texts[i].split())
        target_words.append([target_texts[i].split()])
    
    bleu_1 = corpus_bleu(target_words, output_words, smoothing_function=SmoothingFunction().method0, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(target_words, output_words, smoothing_function=SmoothingFunction().method0, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(target_words, output_words, smoothing_function=SmoothingFunction().method0, weights=(0.33, 0.33, 0.33, 0))
    return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3}


def average_text_lengths(output_texts):
    return {"avg length": sum([len(t.split()) for t in output_texts]) / len(output_texts)}



