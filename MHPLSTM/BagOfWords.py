from sklearn.feature_extraction.text import CountVectorizer
import torch


def build_vectorizer_and_vocab(sentences):
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), max_features=10000, vocabulary=vocab)
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences)
    vocab = vectorizer.vocabulary_
    vocab_size = len(vectorizer.get_feature_names())

    return vectorizer, vocab, vocab_size


def bag_of_words_presentation(sentence, vectorizer, vocab_size):
    sentence = sentence.split()
    outputs = torch.zeros(1, vocab_size)

    for i in range(1, vocab_size):
        input = [' '.join(sentence[:i])]
        output = torch.tensor(vectorizer.transform(input).toarray())
        outputs = torch.cat((outputs, output), dim=0)

    return outputs