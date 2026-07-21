Standard Tiny Shakespeare corpus
================================

`input.txt` is the standard Tiny Shakespeare corpus commonly used by
Karpathy's char-rnn examples.

Source:
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

The v14 synthetic counting experiment tokenizes this file at the character
level, samples contiguous character windows as the haystack, and then replaces
random positions with marker tokens.
