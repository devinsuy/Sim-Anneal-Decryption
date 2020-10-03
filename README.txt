---------
Devin Suy
---------


Input/Output
------------
	- Path directories are listed at top of .py file, modify as necessary
	- Brute Decryption "Sim_Anneal_Decryption.py" reads from: "input_pt1.txt", "n_gram_freq_pt1.txt", and writes to: "output_pt1.txt"
	- Symmetric_Crypto reads from: "input_pt2.txt" and specifies the encryption key at the top of the file that can be modified
	- Execution details are also logged to console
	

Brute Decryption
----------------
	- Uses Mayzner data set available here: http://norvig.com/mayzner.html
	- Raw data was transformed into frequency percentages for 2-grams, 3-grams, ... 9-grams and written to "n_gram_freq_pt1.txt"

	- Frequency percentage is used as metric to create a "decryption score"
		- Where each substring of length n is checked in the corresponding n-gram set -> frequency scores are summed and 
		  assigned to each possible ciphertext decryption

	- English letter frequency data available here: https://en.wikipedia.org/wiki/Letter_frequency
	- Since ciphertext is English, frequency analysis of letters within ciphertext is used and mapped to corresponding english frequency

	- Using the n-gram decryption score, hill climbing algorithm is implemented using simulated annealing
	- All attempted decryptions are scored, sorted in descending order of score, written to "output_pt1.txt"


	- Other ideas that didn't work (commented out):
		- Looking at most common two and three letter words in english
			- Looking at top 25% of two and three letter words in cipher text
			- Attempt to directly map the two together -> leads to higher scores but ... gibberish
		
		- Ranking system based on number of words found in ciphertext and avg word length:
			- Correctly ranked deciphered texts
			- Performed poorly if a bad decision path was taken early on
			- English data set used: https://www.kaggle.com/rtatman/english-word-frequency
		
		- Using the top 50,000 most common words in english language:
			- Used to build keyword mappings to guess if ciphertext was encrypted using one of those words
			- Tried again exhaustively with EVERY word in english language
				- May work for other decryptions but failed for provided sample ciphertexts
	

---------------------------------------------------------------------------------------------------------------------------------------------


Encryption/Decryption
---------------------
	- Creates Crypto class that implements utility methods for Encrypt and Decrypt subclasses
	- Uses the key: "Giacalone" to encrypt messages by generating the shifted alphabet mapping
			Crypto mapping generated with key "Giacalone":
			{
				'A': 'G', 'B': 'I', 'C': 'A', 'D': 'C', 'E': 'L', 'F': 'O', 'G': 'N', 
				'H': 'E', 'I': 'B', 'J': 'D', 'K': 'F', 'L': 'H', 'M': 'J', 'N': 'K', 
				'O': 'M', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 
				'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'
			}
	- Reads text from "input_pt2.txt" by default but messages can be passed as argument
	  to the decrypt or encrypt functions which return the corresponding decrypted/encrypted message
	  