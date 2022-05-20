from sklearn.feature_extraction.text import CountVectorizer
import torch

# vectorizer = CountVectorizer(stop_words='english', ngram_range=(2,2))
def build_vectorizer_and_vocab(sentences):
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences)
    vocab_size = len(vectorizer.get_feature_names())

    return vectorizer, vocab_size


# vectorizer, vocab_size = build_vectorizer_and_vocab(['Game of Thrones is an amazing tv series!', 'Game of Thrones is the best tv series!', 'Game of Thrones is so great'])


def bag_of_words_presentation(sentence, vectorizer, vocab_size):
    sentence = sentence.split()
    outputs = [torch.zeros(vocab_size)]

    for i in range(1, vocab_size):
        input = [' '.join(sentence[:i])]
        output = vectorizer.transform(input).toarray()
        outputs.append(torch.tensor(output))

    return outputs


# print(bag_of_words_presentation('Game of Thrones is an amazing tv series!', vectorizer, vocab_size))