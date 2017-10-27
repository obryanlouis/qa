"""Functions for working with sentences in the dataset.
"""

def find_question_sentence(qst, vocab, example_idx):
    """Gets the question sentence from the question dataset at the given index.
    """
    word_list = []
    for z in range(qst.shape[1]):
        word_id = qst[example_idx, z]
        if vocab.is_pad_word_id(word_id):
            break
        word_list.append(vocab.get_word_for_id(word_id))
    return ' '.join(word_list)
