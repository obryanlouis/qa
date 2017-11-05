"""Class used to generate integer ids for a set of strings where identical
strings get the same id.
"""

class StringCategory():
    def __init__(self):
        self.next_id = 0
        self.categories = {}

    def get_num_categories(self):
        return len(self.categories)

    def get_id_for_word(self, word):
        if word not in self.categories:
            self.categories[word] = self.next_id
            self.next_id += 1
        return self.categories[word]
