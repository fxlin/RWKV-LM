from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

'''
'\nUniversity of Virginia is'

# orig
(Pdb) self.tokenizer.encode(x)
[11, 60918, 4706, 55403, 4600]

# HF tokenizer
(Pdb) self.tokenizer.encode(x).ids
[60918, 2090, 0, 1955]

problme is that HF tokenizer does not split a long string. it only takes one token at
a time(?). if uses space split, then white spaces are dropped. 
(but whitespace is part of the token -- bad)
'''
# Load your vocabulary
vocab = {}
with open("rwkv_vocab_v20230424.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

for l in lines:
    idx = int(l[:l.index(' ')])
    x = eval(l[l.index(' '):l.rindex(' ')])
    x = x.encode("utf-8") if isinstance(x, str) else x
    assert isinstance(x, bytes)
    assert len(x) == int(l[l.rindex(' '):])

    # Convert bytes to string if possible, otherwise use a hexadecimal representation
    try:
        x_str = x.decode("utf-8")
    except UnicodeDecodeError:
        # If decoding fails, convert the bytes to a hex string representation
        # x_str = x.hex()
        x_str = "hex_" + x.hex()


    vocab[x_str] = int(idx)

# Add the <unk> token to the vocabulary with a unique ID
# vocab["<unk>"] = len(vocab)  # Assigning a unique ID at the end of the vocabulary
# vocab["<unk>"] = 65530  # Assigning a unique ID at the end of the vocabulary
vocab["<unk>"] = 0  # Assigning a unique ID at the end of the vocabulary

# Initialize a WordLevel tokenizer with your vocab
tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))

tokenizer.pre_tokenizer = Whitespace()

# Save the tokenizer in JSON format
tokenizer.save("rwkv_vocab_v20230424.json")
