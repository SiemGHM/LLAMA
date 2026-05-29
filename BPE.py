corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]


import collections

class BPETokenizer:
    def __init__(self, data):
        self.data = data
        self.vocab = set( "</w>")
        word_splits = {}
        end_token = "</w>"
        
        
        for doc in self.data:
            words  = doc.split(' ')
            for word in words:
                if word:
                    char_list = list(word) + [end_token]
                    word_tuple = tuple(char_list)
                    
                    if word_tuple not in word_splits:
                        word_splits[word_tuple] = 0
                    word_splits[word_tuple] += 1
                    for char in word_tuple: 
                        self.vocab.add(char)

        print(f"Initial vocab size: {len(self.vocab)}")
        print(f"Initial vocab: {self.vocab}")
        print(f"Initial word splits: {word_splits}")
#         self.token_freq = {}
#         self.token_to_id = {}
#         self.id_to_token = {}
        # self.build_vocab()
        
        
    def get_pair_stats(self, word_splits):
        pair_count = collections.defaultdict(int)
        for word_tuple, freq in word_splits.items():
            symbols = list(word_tuple)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_count[pair] += freq
        return pair_count
    
    def merge_pair(self, word_splits, pair):
        new_word_splits = {}
        (first, second) = pair
        merged_symbol = first + second
        for word_tuple, freq in word_splits.items():
            symbols = list(word_tuple)
            i = 0
            new_symbols = []
        
            while i < len(word_tuple):
                if i< len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(merged_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word_splits[tuple(new_symbols)] = freq
        return new_word_splits
        
    # def build_vocab(self):
    #     for sentence in self.data:
    #         while len(self.vocab) < 1000:  # Example condition, replace with actual BPE logic
    #             pairs = self.get_pairs(sentence)
    #             if not pairs:
    #                 break
    #             best_pair = max(pairs, key=pairs.get)
    #             self.vocab.add(best_pair)
    #             self.update_sentence(sentence, best_pair)   
                
    # def get_pair_s(self, sentence):
    #     pairs = {}
    #     for i in range(len(sentence) - 1):
    #         pair = (sentence[i], sentence[i + 1])
    #         if pair in pairs:
    #             pairs[pair] += 1
    #         else:
    #             pairs[pair] = 1
    #     return pairs
    
    
tokenizer = BPETokenizer(corpus)
