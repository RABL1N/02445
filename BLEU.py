#Import the word_tokenize and sentence_bleu methods from NLTK.
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import nltk
nltk.download('punkt')

#Enter the machine-translated sentence (hypothesis) and the human reference translation (reference).
hypothesis = input("\nPlease enter a machine-translated sentence and confirm with Enter: ")
reference = input("\nPlease enter the corresponding human reference translation and confirm with Enter: ")

#Tokenize the two sentences and store the individual tokes in two lists (one for the hypothesis tokens and one for the reference tokens).
hypothesis = word_tokenize(hypothesis)
reference = word_tokenize(reference)

#Calculate and print BLEU-1, using 1-grams as highest-order n-grams 
#Reference is placed in [square brackets] because you can score the machine-translated sentence against multiple references (applies to all BLEU score calculations listed here).
#The weights are set so that calculation is based solely on 1-gram precision.
BLEU_1 = sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
print(f"\nThe score for BLEU-1 is: {BLEU_1}")

#Calculate and print BLEU-2, using 2-grams as highest-order n-grams.
#The weights are set so that calculation is based on 1-gram and 2-gram precisions.
BLEU_2 = sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0))
print(f"\nThe score for BLEU-2 is: {BLEU_2}")

#Calculate and print BLEU-3, using 3-grams as highest-order n-grams.
#The weights are set so that calculation is based on 1-gram, 2-gram and 3-gram precisions.
BLEU_3 = sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0))
print(f"\nThe score for BLEU-3 is: {BLEU_3}")

#Calculate and print BLEU-4, using 4-grams as highest-order n-grams.
#The weights are set so that calculation is based on 1-gram, 2-gram, 3-gram and 4-gram precisions.
BLEU_4 = sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
print(f"\nThe score for BLEU-4 is: {BLEU_4}")

#Calculate and print BLEU-5, using 5-grams as highest-order n-grams.
#The weights are set so that calculation is based on 1-gram, 2-gram, 3-gram, 4-gram and 5-gram precisions.
BLEU_5 = sentence_bleu([reference], hypothesis, weights=(0.2, 0.2, 0.2, 0.2, 0.2))
print(f"\nThe score for BLEU-5 is: {BLEU_5}")