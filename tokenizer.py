from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from pathlib import Path
import pandas as pd
from transformers import PreTrainedTokenizerFast











# lets get the data in here:
df = pd.read_csv(Path('data/all_posts.csv'))
# convert to list of strings, ensuring compatibility with tokenizer trainer

training_corpus = df['content'].tolist()

# building a WordPiece tokenizer from scratch based on: https://huggingface.co/learn/llm-course/en/chapter6/8
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)

tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)

special_tokens = ["[UNK]", "[PAD]", "[CLS]"]

trainer = trainers.WordPieceTrainer(vocab_size=20000, special_tokens=special_tokens)

tokenizer.train_from_iterator(training_corpus, trainer=trainer)

# post-processing, adding cls token to start of each input sequence
cls_token_id = tokenizer.token_to_id("[CLS]")

tokenizer.post_processor = processors.TemplateProcessing(
    # prepends the input sequence with the cls token
    single=f"[CLS]:0 $A:0",
    special_tokens=[("[CLS]", cls_token_id)],
)

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]"
)


hf_tokenizer.save_pretrained('wp_tokenizer')