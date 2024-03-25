# calculate bleu score from scratch

from typing import List, Tuple
from collections import Counter
from math import exp, log

def n_gram(sentence: List[str], n: int) -> List[Tuple[str]]:
    return [tuple(sentence[i: i + n]) for i in range(len(sentence) - n + 1)]

def clipped_n_gram_counts(candidate: List[str], reference: List[str], n: int) -> Counter:
    candidate_n_grams = n_gram(candidate, n)
    reference_n_grams = n_gram(reference, n)
    reference_n_gram_counts = Counter(reference_n_grams)
    clipped_counts = {
        n_gram: min(count, reference_n_gram_counts[n_gram])
        for n_gram, count in Counter(candidate_n_grams).items()
    }
    return Counter(clipped_counts)

def bleu_score(candidate: List[str], reference: List[str], weights: List[float], n: int = 4) -> float:
    clipped_counts = [clipped_n_gram_counts(candidate, reference, i) for i in range(1, n + 1)]
    precision = [
        sum(clipped_counts[i].values()) / max(1, sum(Counter(n_gram(candidate, i)).values()))
        for i in range(1, n + 1)
    ]
    precision = [precision[i] for i in range(n) if precision[i] > 0]
    if not precision:
        return 0
    precision = [precision[i] for i in range(n) if precision[i] > 0]
    return exp(sum(weights[i] * log(p) for i, p in enumerate(precision)) / len(precision))

def bleu(corpus: List[str], references: List[str], weights: List[float], n: int = 4) -> float:
    return sum(
        bleu_score(candidate.split(), reference.split(), weights, n)
        for candidate, reference in zip(corpus, references)
    ) / len(corpus)
    