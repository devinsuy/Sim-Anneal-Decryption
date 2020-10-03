# ---------
# Devin Suy
# ---------

from pathlib import Path
import sys

# Uses current working directory, modify file "input.txt" 
# contents or Path directory as necessary
in_file = Path("input_pt2.txt")

# Modify decrypt key used
key = "Giacalone"

# Ascii value for alphabet range
UPPER_A = 65
UPPER_Z = 90
LOWER_A = 97
LOWER_Z = 122

# Base class with utility methods
class Crypto:
    # Read text from input, discarding white space and punctuation
    def read_input(self):
        raw = "".join(open(self.in_file).read().split()) 
        chars = [letter for letter in raw if letter.isalpha()]
        return "".join(chars)

    # Generates the alphabet mapping based on a given keyword
    def get_mapping(self, key):
        mapped_letters = set([])
        letter_order = []

        # First append the distinct letters of the word
        # in the word that they appear
        for letter in key:
            letter = letter.upper()
            if letter not in mapped_letters:
                letter_order.append(letter)
                mapped_letters.add(letter)

        # Append the remaining distinct letters of the
        # alphabet in A->Z ordering
        for letter in self.alphabet:
            if letter not in mapped_letters:
                letter_order.append(letter)
                mapped_letters.add(letter)

        # Create mapping in the order we built
        new_map = {}
        for i in range(len(letter_order)):
            new_map[self.alphabet[i]] = letter_order[i]

        # print("Crypto mapping generated with key \"" + str(key) + "\":")
        # print(new_map)
        return new_map
    
    def strip_msg(self, msg):
        return "".join([letter for letter in msg if letter.isalpha()])

    def __init__(self, key, in_file=in_file):
        self.in_file = in_file
        self.alphabet = [chr(ASCII_VAL) for ASCII_VAL in range(UPPER_A, UPPER_Z + 1)]
        self.text = self.read_input()
        self.mapping = self.get_mapping(key)

    # Given an alphabet mapping, replaces each letter with the letter that it maps to
    def replace(self, letter_mapping, msg):
        # Replace each letter according to the given mapping
        new_text = [None] * len(msg)
        for i, letter in enumerate(msg):
            if letter.islower():
                new_text[i] = letter_mapping[letter.upper()].lower()
            else:
                new_text[i] = letter_mapping[letter]

        return "".join(new_text)


class Encrypt(Crypto):
    def __init__(self, key=key, in_file=in_file): 
        Crypto.__init__(self, key, in_file)

    # Uses the key defined at top of file to create alphabet mapping,
    # mapping is used to transform message accordingly and returns it
    def encrypt(self, message=None):
        # Read from input file if not passed a message
        if message is None: message = self.text
        else: message = self.strip_msg(message)
        
        print("Encrypting\n----------")
        print("   Initial Message:", message)
        encrypted_msg = self.replace(self.mapping, message)
        print("   Encrypted:", encrypted_msg)

        return encrypted_msg
        

class Decrypt(Crypto):
    def __init__(self, in_file=in_file): 
        Crypto.__init__(self, key, in_file)
        # Build inverse leter mapping for decryption
        self.decrypt_map = {val: key for key, val in self.mapping.items()}

    # Uses the key defined at top of file to create alphabet mapping,
    # mapping is used to transform message accordingly and returns it
    def decrypt(self, message=None):
        # Read from input file if not passed a message
        if message is None: message = self.text
        else: message = self.strip_msg(message)
        
        print("Decrypting\n----------")
        print("   Initial Message:", message)
        decrypted_msg = self.replace(self.decrypt_map, message)
        print("   Decrypted:", decrypted_msg)

        return decrypted_msg


# Symmetric encryption, key generates mapping 
# used by both encryption and decryption
e = Encrypt()
d = Decrypt()
d.decrypt(e.encrypt())
