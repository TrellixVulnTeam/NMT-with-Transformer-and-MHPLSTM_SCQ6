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


def bag_of_words_presentation(sentence, vectorizer, vocab_size):
    i_vectors = []
    s_vectors = []
    s_ith = torch.zeros(1, vocab_size)
    s_vectors.append(s_ith)
    sentence = sentence.split()

    for i in range(len(sentence)):
        input = [sentence[i]]
        input_ith = torch.tensor(vectorizer.transform(input).toarray(), dtype=torch.float32)
        i_vectors.append(input_ith)
        if (i != 0):
            s_ith = torch.add(s_ith, i_vectors[i-1])
            s_vectors.append(s_ith)

    return i_vectors, s_vectors


# def bag_of_words_presentation(sentence, vectorizer, vocab, vocab_size):
#     sentence = sentence.split()

#     outputs = [0] * vocab_size
#     input_step_0 = re.sub(r'[^\w\s]', '', sentence[0]).lower()

#     if input_step_0 in vocab:
#         input_step_0_idx = vocab[input_step_0]
#     else:
#         input_step_0_idx = len(vocab)
        
#     outputs.append(float(input_step_0_idx))
#     outputs = torch.tensor(outputs).reshape((1, vocab_size + 1))

#     for i in range(1, len(sentence)):
#         input = [' '.join(sentence[:i])]
#         output = vectorizer.transform(input).toarray()
#         input_step_t = sentence[i]
#         input_step_t = re.sub(r'[^\w\s]', '', input_step_t).lower()

#         if input_step_t in vocab:
#             input_step_t_idx = vocab[input_step_t]
#         else:
#             input_step_t_idx = len(vocab)

#         output = np.append(output, float(input_step_t_idx))
#         output = torch.tensor(output).reshape((1, vocab_size + 1))
#         outputs = torch.cat((outputs, output), dim=0)

#     return outputs


class BagOfWord(nn.Module):
    """Preprocess data"""
    def __init__(self, sentences):
        super(BagOfWord, self).__init__()
        self.vectorizer, self.vocab, self.vocab_size = build_vectorizer_and_vocab(sentences)

    def forward(self, input):
        return bag_of_words_presentation(input, self.vectorizer, self.vocab_size)