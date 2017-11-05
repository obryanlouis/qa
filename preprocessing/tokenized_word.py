"""Representation for a tokenized word within a sentence.
"""

class TokenizedWord():
    def __init__(self, word, start, end, ner, pos):
        self.word = word
        self.start = start
        self.end = end
        # Named entity.
        self.ner = ner
        # Part of speech.
        self.pos = pos

    def __repr__(self):
        return "word: "  + str(self.word) \
            + " start: " + str(self.start) \
            + " end: "   + str(self.end) \
            + " ner: "   + str(self.ner) \
            + " pos: "   + str(self.pos)
