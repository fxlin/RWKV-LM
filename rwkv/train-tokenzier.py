from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize a BPE tokenizer
tokenizer = Tokenizer(BPE())

# Use whitespace as a pre-tokenizer
tokenizer.pre_tokenizer = Whitespace()

# Define the trainer
trainer = BpeTrainer(special_tokens=["<unk>", "<pad>", "<s>", "</s>"], vocab_size=65536)

# Train the tokenizer using your vocabulary file (plain text file)
files = ["rwkv_vocab_v20230424.txt"]  # Replace with your vocabulary or text corpus file
tokenizer.train(files, trainer)

# Save the tokenizer in JSON format
tokenizer.save("rwkv_vocab_v20230424.json")
