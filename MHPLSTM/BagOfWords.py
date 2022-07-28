from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn


def build_vectorizer_and_vocab(sentences):
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), max_features=10000, vocabulary=vocab)
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences)
    vocab = vectorizer.vocabulary_
    vocab_size = len(vectorizer.get_feature_names())

    return vectorizer, vocab, vocab_size


def bag_of_words_presentation(sentence, vectorizer):
    i_vectors = []
    sentence = sentence.split()

    for i in range(len(sentence)):
        input = [sentence[i]]
        input_ith = torch.tensor(vectorizer.transform(input).toarray(), dtype=torch.float32)
        i_vectors.append(input_ith)

    i_vectors = torch.stack([i_vector.squeeze() for i_vector in i_vectors])

    return i_vectors


class BagOfWord(nn.Module):
    """Preprocess data"""
    def __init__(self, sentences):
        super(BagOfWord, self).__init__()
        self.vectorizer, self.vocab, self.vocab_size = build_vectorizer_and_vocab(sentences)

    def forward(self, input):
        return bag_of_words_presentation(input, self.vectorizer)