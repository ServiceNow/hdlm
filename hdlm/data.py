import re
from transformers import GPT2TokenizerFast
import datasets
from itertools import chain
import numpy as np

import requests
import json
from datasets import Dataset

from torch.utils.data import DataLoader, DistributedSampler


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()



def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=40, debug=False, efficient_annealing=False):
    if name == "wikitext103":
        dataset = datasets.load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = datasets.load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = datasets.load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == 'openwebtext-valid':
        dataset = datasets.load_dataset('openwebtext', split='train[-100000:]', cache_dir=cache_dir, trust_remote_code=True)
    elif name == 'openwebtext-train':
        dataset = datasets.load_dataset('openwebtext', split='train[:-100000]', cache_dir=cache_dir, trust_remote_code=True)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    elif name == 'scientific_papers_arxiv':
        dataset = datasets.load_dataset('scientific_papers', 'arxiv', trust_remote_code=True, cache_dir=cache_dir)
    elif name == 'scientific_papers_pubmed':
        dataset = datasets.load_dataset('scientific_papers', 'pubmed', trust_remote_code=True, cache_dir=cache_dir)
    elif name == 'ag_news':
        dataset = datasets.load_dataset('ag_news',cache_dir=cache_dir)
    else:
        dataset = datasets.load_dataset(name, cache_dir=cache_dir)

    if name in {"lambada", "openwebtext-valid", "openwebtext-train"}:
        data = dataset
    else:
        data = dataset[mode]
    # Add debug mode: Use only 1% of the dataset
    if debug:
        total_examples = len(data)
        debug_size = max(1, int(0.02 * total_examples))
        data = data.select(range(debug_size))
        print(f"Debug mode: Using {debug_size}/{total_examples} examples ({(100 * debug_size / total_examples):.2f}%)")

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name.startswith("scientific_papers"):
        detokenizer = scientific_papers_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example['sentence']
        elif 'scientific_papers' in name:
            text = example['article']
        else:
            text = example["text"]
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following 
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    elif 'scientific_papers' in name:
        tokenized_dataset = tokenized_dataset.remove_columns(['article', 'abstract', 'section_names'])
    elif name == 'ag_news':
        tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label'])
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    # Add positions to the dataset
    def add_positions(examples):
        # examples['input_ids'] is a list of lists of token ids
        positions = [list(range(len(ids))) for ids in examples['input_ids']]
        examples['positions'] = positions
        return examples

    chunked_dataset = chunked_dataset.map(
        add_positions,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True
    )

    if efficient_annealing:
        # Function to duplicate sequences and positions
        def duplicate_sequences(examples):
            examples['input_ids'] = [ids + ids for ids in examples['input_ids']]
            examples['positions'] = [pos + pos for pos in examples['positions']]
            return examples
        # Apply the duplication to the dataset
        chunked_dataset = chunked_dataset.map(
            duplicate_sequences,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True
        )
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset


def get_dataloaders(config, distributed=False, train=True):
    debug = config.data.debug if "debug" in config.data else False
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    if train:
        train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length, debug=debug, efficient_annealing=config.annealing.efficient)
    valid_set = get_dataset(config.data.valid, "validation" if config.data.valid not in {"text8", "lm1b", "ag_news"} else "test", cache_dir=config.data.cache_dir, block_size=config.model.length, debug=False, efficient_annealing=config.annealing.efficient)

    if distributed:
        if train:
            train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedSampler(valid_set)
    else:
        if train:
            train_sampler = None
        test_sampler = None
    
    if train:
        train_loader = DataLoader(
            train_set,
            batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(train_sampler is None),
            persistent_workers=True,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(test_sampler is None),
        )
    else:
        valid_loader = DataLoader(
            valid_set,
            batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(test_sampler is None),
        )
    if train:
        return train_loader, valid_loader
    else:
        return valid_loader