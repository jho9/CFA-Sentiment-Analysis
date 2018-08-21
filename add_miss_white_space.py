# This is a fork off of a design from another individual who attempted to split strings that were missing white space
# In order to do so we modified an accounted for Sentences which contained whitespaces in order to process the string properly. 
from math import log
# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open("C:\\Users\\James\\Documents\\wordbank.txt").read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

# Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
def best_match(i):
    candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
    return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""
    
    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))

def infer_sent_whitespace(s):
    s = s.lower()
    l = s.split()
    j = []
    for i in range(0,len(l)):
        if l[i][0] == '@':
            j.append(l[i])
        elif l[i][-1] in ['.' , '?' , '!' , ',' , ';', ':']:
            j.append(l[i])
        else:
            j.append(infer_spaces(l[i]))
    
    return ' '.join(j)
