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
        self.vocab = {"</w>"}
        self.merges = []
        self.word_splits = {}
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
                    self.word_splits = word_splits
                    for char in word_tuple: 
                        self.vocab.add(char)

        print(f"Initial vocab size: {len(self.vocab)}")
        print(f"Initial vocab: {self.vocab}")
        print(f"Initial word splits: {word_splits}")
        num_merges = 10
        for i in range(num_merges):
            pair_stats = self.get_pair_stats(word_splits)
            if not pair_stats:
                break
            best_pair = max(pair_stats, key=pair_stats.get)
            self.merges.append(best_pair)
            word_splits = self.merge_pair(word_splits, best_pair)
            self.word_splits = word_splits
            self.vocab.add(''.join(best_pair))
            print(f"Merge {i + 1}: Merged pair {best_pair} into {''.join(best_pair)}")
            print(f"Updated vocab size: {len(self.vocab)}")
            print(f"Updated vocab: {self.vocab}")
            print(f"Merge history: {self.merges}")
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
        new_word_splits = collections.defaultdict(int)
        (first, second) = pair
        merged_symbol = first + second

        for word_tuple, freq in word_splits.items():
            symbols = list(word_tuple)
            i = 0
            new_symbols = []

            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(merged_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            new_word_splits[tuple(new_symbols)] += freq

        return dict(new_word_splits)
    
    
    def tokenize(self, text):
        words = text.split(' ')
        tokens = []
        for word in words:
            if word:
                # start with character-level symbols (with end-of-word marker)
                symbols = list(word) + ["</w>"]

                # apply stored merges in order (greedy replacement per merge)
                for merge in self.merges:
                    first, second = merge
                    merged = first + second
                    i = 0
                    new_symbols = []
                    while i < len(symbols):
                        if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                            new_symbols.append(merged)
                            i += 2
                        else:
                            new_symbols.append(symbols[i])
                            i += 1
                    symbols = new_symbols

                # append final tokens (flattened)
                tokens.extend(symbols)
        return tokens

    def detokenize(self, tokens):
        # Reconstruct words from tokens using the end-of-word marker
        words = []
        current = ""
        for tok in tokens:
            if isinstance(tok, str) and tok.endswith("</w>"):
                current += tok[:-4]
                words.append(current)
                current = ""
            else:
                current += tok

        # If any trailing partial word (no </w>), append it
        if current:
            words.append(current)

        return " ".join(words)
    
    def encode(self, tokens):
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(token)
            else:
                token_ids.append("<unk>")
        return token_ids
        
    
    
tokenizer = BPETokenizer(corpus)

# --- Simple test: tokenize and detokenize a sentence ---
test_sentence = "This is the second document."
tokens = tokenizer.tokenize(test_sentence)
print('\nTest sentence:', test_sentence)
print('Tokens:', tokens)
detok = tokenizer.detokenize(tokens)
print('Detokenized:', detok)
