# ---------
# Devin Suy
# ---------
from pathlib import Path
from collections import defaultdict, OrderedDict
import math, itertools, random, json, copy
import sys, time

# Uses current working directory, modify file "input.txt" 
# contents or Path directory as necessary
in_file = Path("input_pt1.txt")
out_file = Path("output_pt1.txt")
ngram_file = Path("n_gram_freq_pt1.txt")

# Ascii value for alphabet range
UPPER_A = 65
UPPER_Z = 90
LOWER_A = 97
LOWER_Z = 122

# Adjust simulated annealing temperature range (default 150,000)
max_iters = 150000

# Optimal decryption should MAXIMIZE decryption score, use hill
# climbing approach with simulated annealing
class SimAnneal:
    def __init__(self, init_score, letter_map):
        self.base_score = self.curr_score = init_score
        self.max_iters = max_iters
        self.iter_count = 0
        self.map = letter_map
        self.temp = 1

        print("Simulated annealing init with:")
        print("   Metric @", init_score)

    def update_temp(self):
        ratio = float(self.iter_count / self.max_iters)
        self.iter_count += 1
        self.temp = max(sys.float_info.min, 1-ratio)
    
    def finished(self): return self.temp == sys.float_info.min

    def take_solution(self, score, letter_map):
        if letter_map == self.map or score == self.curr_score: return True
        self.update_temp()
        delta_s = score - self.curr_score

        # Always take a better solution (decyrpt score is higher)
        if delta_s > 0:
            self.curr_score = score
            self.map = letter_map
            return True
        else:
            # Current temperature permits accepting a less optimal
            # letter assignment (explore search space)
            if math.exp(delta_s / self.temp) > random.uniform(0,1):
                self.curr_score = score
                self.map = letter_map
                return True
            else: 
                return False

class Decrypt:
    def load_n_freq(self, n):
        with open(ngram_file, "r") as freq:
            begin = False
            start_line = str(n) + "-gram"
            end_line = str(n+1) + "-gram" 
            
            # Map the n_gram to it's appearence frequency for the scoring metric
            ngram_freq = OrderedDict()
            for line in freq.read().split("\n"):
                if not begin:
                    if line == start_line: begin = True
                else:
                    if line == end_line: break
                    n_gram, freq = line.split(",")
                    ngram_freq[n_gram] = float(freq)
            
            return ngram_freq

    def load_n_grams(self):
        # Load the top 2:9n-grams ranked by descending order of frequency
        # Source: Frequency built from Mayzner raw data set http://norvig.com/mayzner.html
        self.grams_2 = self.load_n_freq(2)
        self.grams_3 = self.load_n_freq(3)
        self.grams_4 = self.load_n_freq(4)
        self.grams_5 = self.load_n_freq(5)
        self.grams_6 = self.load_n_freq(6)
        self.grams_7 = self.load_n_freq(7)
        self.grams_8 = self.load_n_freq(8)
        self.grams_9 = self.load_n_freq(9)

        # Build a n_gram set for O(1) contains() operation
        self.n_grams = set([])
        self.n_grams.update(self.grams_2.keys())
        self.n_grams.update(self.grams_3.keys())
        self.n_grams.update(self.grams_4.keys())
        self.n_grams.update(self.grams_5.keys())
        self.n_grams.update(self.grams_6.keys())
        self.n_grams.update(self.grams_7.keys())
        self.n_grams.update(self.grams_8.keys())
        self.n_grams.update(self.grams_9.keys())

    def __init__(self, in_file=in_file, out_file=out_file):
        self.in_file = in_file
        self.out_file = out_file
        self.load_n_grams()

        # Read ciphertext from input, removing all white spaces
        self.ciphertext = "".join(open(self.in_file).read().split())

        # Premap the frequency rank to the corresponding English letter
        # Source: https://en.wikipedia.org/wiki/Letter_frequency
        self.english_freq = {
            1: 'E', 2: 'T', 3: 'A', 4: 'O', 5: 'I', 6: 'N', 7: 'S', 8: 'H', 9: 'R', 10: 'D',
            11: 'L', 12: 'U', 13: 'C', 14: 'W', 15: 'M', 16: 'F', 17: 'Y', 18: 'G', 19: 'P',
            20: 'B', 21: 'V', 22: 'K', 23: 'X', 24: 'J', 25: 'Q', 26: 'Z'
        }
        self.alphabet = [chr(ASCII_VAL) for ASCII_VAL in range(UPPER_A, UPPER_Z + 1)]

        # Keep track of the highest score found so far and the alphabet mapping used to reach it
        self.best_score = self.best_assignment = self.direct_map = None

        # Maintain a cache mapping (DP) of transformed ciphertexts -> to the decrypt score
        self.score_cache = {}

    # Given an ciphertext, returns all n-gram subtrings
    def get_ngrams(self, cipher_text, n):
        n_grams = []
        for i in range(len(cipher_text)-n+1):
            n_grams.append(cipher_text[i:i+n].upper())
        return n_grams

    # Scores a given ciphertext by it's present n-grams
    # and the corresponding frequency ratings
    def decrypt_score(self, cipher_text):
        if cipher_text in self.score_cache: return self.score_cache[cipher_text]
        score = 0
        for n_gram in self.get_ngrams(cipher_text, 2):
            if n_gram in self.n_grams: score += self.grams_2[n_gram]
        for n_gram in self.get_ngrams(cipher_text, 3):
            if n_gram in self.n_grams: score += self.grams_3[n_gram]
        for n_gram in self.get_ngrams(cipher_text, 4):
            if n_gram in self.n_grams: score += self.grams_4[n_gram]
        for n_gram in self.get_ngrams(cipher_text, 5):
            if n_gram in self.n_grams: score += self.grams_5[n_gram]
        for n_gram in self.get_ngrams(cipher_text, 6):
            if n_gram in self.n_grams: score += self.grams_6[n_gram]
        for n_gram in self.get_ngrams(cipher_text, 7):
            if n_gram in self.n_grams: score += self.grams_7[n_gram]
        for n_gram in self.get_ngrams(cipher_text, 8):
            if n_gram in self.n_grams: score += self.grams_8[n_gram]
        for n_gram in self.get_ngrams(cipher_text, 9):
            if n_gram in self.n_grams: score += self.grams_9[n_gram]
        self.score_cache[cipher_text] = score    

        return score

    # Given a shift value, performs caesar shift on 
    # the ciphertext and returns it
    def shift_line(self, key, line=None):
        if line is None: line = self.ciphertext
        # A 26 shift means no change, values > 26 can
        # be represented same way if we simply mod
        key = key % 26
        if key == 0 or key == 26:
            return line
        shifted = [None] * len(line)
        shift_ptr = 0

        # Perform letter shifting by key letters
        for char in line:
            ASCII_VAL = ord(char)
            SHIFTED_VAL = ASCII_VAL + key

            # Retain the casing of the characters, remaining shift that exceeds
            # letter z should continue past start at letter a
            if char.islower() and SHIFTED_VAL > LOWER_Z:
                shifted_char = chr(LOWER_A + (SHIFTED_VAL - LOWER_Z - 1))
            elif char.isupper() and SHIFTED_VAL > UPPER_Z:
                shifted_char = chr(UPPER_A + (SHIFTED_VAL - UPPER_Z - 1))
            else:
                shifted_char = chr(SHIFTED_VAL)

            shifted[shift_ptr] = shifted_char
            shift_ptr += 1

        return "".join(shifted)

    # Given an alphabet mapping, replaces each letter with the letter that it maps to
    def replace(self, letter_mapping, ciphertext=None):
        if ciphertext is None:
            ciphertext = self.ciphertext

        # Replace each letter according to the given mapping
        new_text = [None] * len(ciphertext)
        for i, letter in enumerate(ciphertext):
            if letter.islower():
                new_text[i] = letter_mapping[letter.upper()].lower()
            else:
                new_text[i] = letter_mapping[letter]

        return "".join(new_text)

    # Performs a caesar shift for values 1:25, returning a
    # set of all possible shifted ciphertexts
    def get_caesar_set(self):
        shifted = set([])
        for shift_val in range(1, 26):
            shifted.add(self.shift_line(key=shift_val))

        return shifted


    # Scan all substrings of each line of the ciphertext
    # to obtain a set of all english words present
    # def check_for_words(self, ciphertext):
    #     # Avoid recalculating, lookup result if present
    #     if ciphertext in self.score_cache:
    #         return self.score_cache[ciphertext]

    #     words_found = []
    #     indicies_used = set([])
    #     for i, letter in enumerate(ciphertext):
    #         for j in range(i + 1, len(ciphertext)):
    #             substr = ciphertext[i:j + 1]
    #             if substr in self.word_set: 
    #                 words_found.append(substr)
    #                 for used_index in range(i, j+1): indicies_used.add(used_index)
        
    #     if len(words_found) == 0:
    #         return {'avg_word_len': 0, 'longest_word': 0, 'max_word_len': 0, 'util_ratio': 0}

    #     # Determine the length of the longest word found
    #     max_len = 0
    #     longest_word = None
    #     for word in words_found:
    #         if len(word) > max_len: 
    #             max_len = len(word)
    #             longest_word = word

    #     # Get the average length of words in the string
    #     total_len = 0
    #     for word in words_found: total_len += len(word)
    #     avg_len = total_len / len(words_found)   

    #     ctext_data = {
    #         'avg_word_len': avg_len,
    #         'longest_word' : longest_word,
    #         'max_word_len': max_len,
    #         'util_ratio': len(indicies_used) / len(self.ciphertext)
    #         # 'words' : words_found        
    #     }
    #     # Cache values for future lookup
    #     self.score_cache[ciphertext] = ctext_data

    #     return ctext_data

    # Checks the frequency of all two and three letter substrings
    # of the ciphertext, builds lists in descending order of frequency
    # def set_di_tri(self):
    #     # Map all possible digraph substrings to their frequency in the ciphertext
    #     di_freq = defaultdict(int)
    #     for i in range(len(self.ciphertext) - 1):
    #         di_graph = self.ciphertext[i:i + 2]
    #         di_freq[di_graph] += 1

    #     # Consider only digraphs in the upper 75% of frequency that also appear atleast twice
    #     stop_index = int(len(di_freq) * 0.25)
    #     self.di_graphs = [
    #                          di_graph.upper() for di_graph in sorted(di_freq, key=di_freq.__getitem__, reverse=True) 
    #                          if di_freq[di_graph] > 1
    #                      ][:stop_index]

    #     # Repeat process for trigraphs
    #     tri_freq = defaultdict(int)
    #     for i in range(len(self.ciphertext) - 2):
    #         tri_graph = self.ciphertext[i:i + 3]
    #         tri_freq[tri_graph] += 1

    #     stop_index = int(len(tri_freq) * 0.25)
    #     self.tri_graphs = [
    #                           tri_graph.upper() for tri_graph in sorted(tri_freq, key=tri_freq.__getitem__, reverse=True) 
    #                           if tri_freq[tri_graph] > 1
    #                       ][:stop_index]

    # get_di_set() utility function
    # def di_assignments(self, assignments, chaining=False, reverse=False):
    #     di_set = set([])
    #     if reverse:
    #         assignments = reversed(assignments)

    #     for curr_assignment in assignments:
    #         if chaining == False:
    #             new_map = copy.deepcopy(self.direct_map)
    #         else:
    #             new_map = copy.deepcopy(self.best_assignment)
    #         curr_letter_1, curr_letter_2 = curr_assignment[0]
    #         goal_letter_1, goal_letter_2 = curr_assignment[1]

    #         # Locate the letters that currently map to our goal letters
    #         for letter, mapped_letter in new_map.items():
    #             if mapped_letter == goal_letter_1: replaced_letter_1 = letter
    #             if mapped_letter == goal_letter_2: replaced_letter_2 = letter

    #         # Only process complete mappings
    #         try:
    #             if replaced_letter_1 is not None and replaced_letter_2 is not None :
    #                 pass
    #         except:
    #             continue


    #         # Swap the mappings of the letters
    #         new_map[replaced_letter_1] = new_map[curr_letter_1]
    #         new_map[replaced_letter_2] = new_map[curr_letter_2]
    #         new_map[curr_letter_1] = goal_letter_1
    #         new_map[curr_letter_2] = goal_letter_2

    #         # Save the mapping, overwrite if better than previous best
    #         new_ci_text = self.replace(new_map)
    #         decrypt_score = self.decrypt_score(new_ci_text)
    #         di_set.add(new_ci_text)

    #         if decrypt_score > self.best_score:
    #             print("   Better found with score", self.best_score, "->", decrypt_score)
    #             self.best_assignment = copy.deepcopy(new_map)
    #             self.best_score = decrypt_score

    #     return di_set

    # Consider permutations of top english Digraphs and top Digraphs found
    # in the ciphertext, create mappings that attempt to match them
    # def get_di_set(self):
    #     print("\nMerging With Digraphs\n---------------------")
    #     print("Current metric @", self.best_score)

    #     di = [self.di_graphs, list(self.grams_2.keys())[:len(self.grams_2)//2]]
    #     assignments = list(itertools.product(*di))
    #     di_set = set([])

    #     # Consider digraph swaps independent of one another 
    #     # (all start from direct mapping)
    #     di_set = di_set.union(self.di_assignments(assignments))

    #     # Consider chaining assignments, also right to left ordering
    #     di_set = di_set.union((self.di_assignments(assignments, chaining=True)))
    #     di_set = di_set.union((self.di_assignments(assignments, chaining=True, reverse=True)))

    #     return di_set

    # # get_tri_set() utility function
    # def tri_assignments(self, assignments, chaining=False, reverse=False):
    #     tri_set = set([])
    #     if reverse:
    #         assignments = reversed(assignments)

    #     for curr_assignment in assignments:
    #         if chaining == False:
    #             new_map = copy.deepcopy(self.direct_map)
    #         else:
    #             new_map = copy.deepcopy(self.best_assignment)
    #         curr_letter_1, curr_letter_2, curr_letter_3 = curr_assignment[0]
    #         goal_letter_1, goal_letter_2, goal_letter_3 = curr_assignment[1]

    #         # Locate the letters that currently map to our goal letters
    #         for letter, mapped_letter in new_map.items():
    #             if mapped_letter == goal_letter_1: replaced_letter_1 = letter
    #             if mapped_letter == goal_letter_2: replaced_letter_2 = letter
    #             if mapped_letter == goal_letter_3: replaced_letter_3 = letter

    #         # Only process complete mappings
    #         try:
    #             if replaced_letter_1 is not None and replaced_letter_2 is not None and replaced_letter_3 is not None:
    #                 pass
    #         except:
    #             continue

    #         # Swap the mappings of the letters
    #         new_map[replaced_letter_1] = new_map[curr_letter_1]
    #         new_map[replaced_letter_2] = new_map[curr_letter_2]
    #         new_map[replaced_letter_3] = new_map[curr_letter_3]
    #         new_map[curr_letter_1] = goal_letter_1
    #         new_map[curr_letter_2] = goal_letter_2
    #         new_map[curr_letter_3] = goal_letter_3
            
    #         # Save the mapping, overwrite if better than previous best
    #         new_ci_text = self.replace(new_map)
    #         decrypt_score = self.decrypt_score(new_ci_text)
    #         tri_set.add(new_ci_text)

    #         if decrypt_score > self.best_score:
    #             print("   Better found with score", self.best_score, "->", decrypt_score)
    #             self.best_assignment = copy.deepcopy(new_map)
    #             self.best_score = decrypt_score

    #     return tri_set

    # Consider permutations of top english TRIgraphs and top TRIgraphs found
    # in the ciphertext, create mappings that attempt to match them
    # def get_tri_set(self):
    #     print("\nMerging With Trigraphs\n----------------------")
    #     print("Current metric @", self.best_score)

    #     tri = [self.tri_graphs, list(self.grams_3.keys())[:len(self.grams_3)//2]]
    #     assignments = list(itertools.product(*tri))
    #     tri_set = set([])

    #     # Consider trigraph swaps independent of one another 
    #     # (all start from direct mapping)
    #     tri_set = tri_set.union(self.tri_assignments(assignments))

    #     # Consider chaining assignments, also right to left ordering
    #     tri_set = tri_set.union((self.tri_assignments(assignments, chaining=True)))
    #     tri_set = tri_set.union((self.tri_assignments(assignments, chaining=True, reverse=True)))

    #     return tri_set

    # Given a keyword return the mapping built from the word
    # def keyword_mapping(self, word):
    #     mapped_letters = set([])
    #     letter_order = []

    #     # First append the distinct letters of the word
    #     # in the word that they appear
    #     for letter in word:
    #         letter = letter.upper()
    #         if letter not in mapped_letters:
    #             letter_order.append(letter)
    #             mapped_letters.add(letter)

    #     # Append the remaining distinct letters of the
    #     # alphabet in A->Z ordering
    #     for letter in self.alphabet:
    #         if letter not in mapped_letters:
    #             letter_order.append(letter)
    #             mapped_letters.add(letter)

    #     # Create mapping in the order we built
    #     new_map = {}
    #     for i in range(len(letter_order)):
    #         new_map[self.alphabet[i]] = letter_order[i]

    #     return new_map


    # Performs direct alphabet mapping based on frequency of letters
    # in ciphertext, matched to corresponding English letter frequency
    def get_frequency_set(self):
        print("Direct Frequency Mapping\n------------------------")
        # Map letters to the frequency found in the ciphertext
        letter_freq = defaultdict(int)
        for letter in self.ciphertext:
            letter_freq[letter.upper()] += 1
        present = set(letter_freq.keys())
        not_present = set(self.alphabet).difference(present)
        freq_set = set()

        # Retrieve the list of letters sorted by descending frequency
        self.rank_list = [letter for letter in sorted(letter_freq, key=letter_freq.__getitem__, reverse=True)]

        # Maps the letter to the correpsondign rank based on the
        # frequency of letters found in the ciphertext
        self.letter_rank = {}
        for rank, letter in enumerate(self.rank_list): self.letter_rank[letter] = rank + 1

        # Build naive direct mapping based on correlation to English frequency
        direct_map = OrderedDict()
        for letter, rank in self.letter_rank.items():
            direct_map[letter] = self.english_freq[rank]
        freq_set.add(self.replace(direct_map))

        self.best_score = self.base_score = self.decrypt_score(self.replace(direct_map))
        self.best_assignment = copy.deepcopy(direct_map)
        self.direct_map = copy.deepcopy(direct_map)

        return freq_set

    # get_swap_set() utility function
    def adj_swap(self, chaining=False):
        letters = list(self.best_assignment.keys())
        swap_set = set([])

        # First letter has no left adjacent neighbor, last has no right
        self.poss_swaps = [
            (letters[0], letters[1]), (letters[-1], letters[-2])
        ]
        # Enumerate all possible swaps
        for i in range(1, len(letters)-1):
            self.poss_swaps.append((letters[i], letters[i-1]))
            self.poss_swaps.append((letters[i], letters[i+1]))
        random.shuffle(self.poss_swaps)
        
        if chaining: src_map = copy.deepcopy(self.best_assignment)
        else: src_map = copy.deepcopy(self.direct_map)

        # Perform multiple passes of all swaps
        for _ in range(1000):
            swap_made = False
            swap_tuples = copy.deepcopy(self.poss_swaps)
            # random.shuffle(swap_tuples)
            while swap_tuples:
                new_map = src_map
                swap_1, swap_2 = swap_tuples.pop()

                # Swap the letter mappings
                temp = new_map[swap_1]
                new_map[swap_1] = new_map[swap_2]
                new_map[swap_2] = temp
                new_ctext = self.replace(new_map)
                decrypt_score = self.decrypt_score(new_ctext)
                swap_set.add(new_ctext)

                if self.anneal.take_solution(decrypt_score, new_map):
                    # print("   Better found with score", self.best_score, "->", decrypt_score)
                    self.best_assignment = new_map
                    self.best_score = decrypt_score
                    swap_made = True
            
            # We have exhausted all possible adjacent swaps
            if not swap_made: 
                break

        return swap_set
    
    # Generates additional ciphertexts by using the current best assignment
    # and swapping mappings with adjacent appearence frequencies,
    # updating the best assignment each time an improvement is found
    def get_swap_set(self, chaining=False, reverse=False):
        print("\nAdjacent Frequency Swapping\n---------------------------")
        print("Current metric @", self.best_score)
        swap_set = set([])
        swap_set = swap_set.union(self.adj_swap())
        swap_set = swap_set.union(self.adj_swap(chaining=True))
        
        return swap_set

    # Randomly selects two letters and swaps their mappings
    def get_rand_set_util(self):
        new_map = copy.deepcopy(self.best_assignment)
        letters = list(self.best_assignment.keys())
        swap_letter_1 = random.choice(letters)
        letters.remove(swap_letter_1)
        swap_letter_2 = random.choice(letters)

        # Swap the letters
        temp = new_map[swap_letter_1]
        new_map[swap_letter_1] = new_map[swap_letter_2]
        new_map[swap_letter_2] = temp

        return new_map

    # Perform random unrestricted mapping swaps, overwriting if
    # finding a better mapping
    def get_rand_set(self):
        print("\nRandom Swapping\n---------------")
        print("Current metric @", self.best_score)
        rand_set = set([])
        
        while not self.anneal.finished():
            rand_map = self.get_rand_set_util()
            new_ctext = self.replace(rand_map)
            decrypt_score = self.decrypt_score(new_ctext)
            rand_set.add(new_ctext)
            if self.anneal.take_solution(decrypt_score, letter_map=rand_map):
                self.best_assignment = rand_map
                self.best_score = decrypt_score
        
        return rand_set

    # Given the output dictionary of all generated variations of the ciphertext
    # sorts each by descending decryption score and writes to output file  
    def write_output(self, output: dict, out_file=out_file):
        ranked_decryptions = []
        for rank, ciphertext in enumerate(sorted(output.items(), key=lambda x: x[1], reverse=True)):
            ranked_decryptions.append((ciphertext, "RANK #" + str(rank+1)))
        open(out_file, "w").write(json.dumps(ranked_decryptions, indent=4))

    def decrypt(self):
        # Maps variations of the cipher text generated -> to 
        # the decryption score
        decryption_scores = {}
        poss_decryptions = set([])

        # Consider the 25 variations of a simple caesar shift independently
        # may not always be ranked BEST solution, check console MANUALLY
        poss_decryptions = poss_decryptions.union(self.get_caesar_set())
        print("Caesar Solutions\n----------------")
        for i in poss_decryptions: print(i,"\n")

        # Consider variations in which frequency of letters found in the cipher
        # text are mapped based on correlation with English letter frequencies
        poss_decryptions = poss_decryptions.union(self.get_frequency_set())

        # Continue to iterate upon the best mapping we've found so far by 
        # swapping mappings of letters with adjacent frequencies
        self.anneal = SimAnneal(init_score=self.best_score, letter_map=self.best_assignment)
        poss_decryptions = poss_decryptions.union(self.get_swap_set())

        # Do not begin randomization until frequency swapping produces a better solution
        if self.anneal.curr_score < self.anneal.base_score:
            while self.anneal.curr_score < self.anneal.base_score:
                print("restarting")
                self.best_score = self.anneal.base_score
                self.best_map = self.direct_map
                self.anneal = SimAnneal(init_score=self.anneal.base_score, letter_map=self.direct_map)
                poss_decryptions = poss_decryptions.union(self.get_swap_set())
        
        # Begin SA random swapping
        poss_decryptions = poss_decryptions.union(self.get_rand_set())
        

        # Correlate highest frequency English digraphs with those most found in
        # the ciphertext, generate assignments by performing letter mapping swaps
        # self.set_di_tri()
        # poss_decryptions = poss_decryptions.union(self.get_di_set())
        
        # Attempt frequency swaps again after merging
        # poss_decryptions = poss_decryptions.union(self.get_swap_set())

        # Repeat for digraph frequencies
        # poss_decryptions = poss_decryptions.union(self.get_tri_set())

        # Attempt frequency swaps again after merging
        # poss_decryptions = poss_decryptions.union(self.get_swap_set())
        
        # Check if any final improvements can be made from random swaps
        # poss_decryptions = poss_decryptions.union(self.get_rand_set())


        # Rank all generated decryptions
        best_decrypt = None
        for decryption in poss_decryptions:
            curr_score = self.decrypt_score(decryption)
            if curr_score >= self.best_score:
                self.best_score = curr_score
                best_decrypt = decryption
            decryption_scores[decryption] = curr_score
        
        print("End metric @", self.best_score)
        print("\nBest decryption:")
        print(best_decrypt)

        # Write results sorted in descending order of decrypted likelihood
        self.write_output(decryption_scores)


d = Decrypt()
d.decrypt()
