import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
import argparse
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
import torch.multiprocessing as mp
from utils import *
from layers import *
from transformer import *

warnings.filterwarnings("ignore")

from torchnlp.datasets import multi30k_dataset

## Create a seq2seq model using the layers and transformer model defined

def create_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # Initialize parameters with Glorot / fan_avg as defined in original paper
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


## Classes and functions useful for training the model

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    
    
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total number of examples used
    tokens: int = 0  # total number of tokens processed
    
    
def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        
        # Forward Pass
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        
        # Compute loss
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train":
            # Backward pass
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            
            # Apply optimizer step every accum_iter iterations
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

    
    
    
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    
    
    
## Load the tokenizer, dataset etc and prepare vocabulary

def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])
        
        
def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    # fall 2023 version notes: the original link for the Multi30K dataset expired so we're using the alternative 
    # dataset link from https://discuss.pytorch.org/t/multi30k-dataset-can-not-load-from-the-torchtext-because-the-main-server-is-down/153657
    train, val, test = multi30k_dataset(
        train=True,
        dev=True,
        test=True,
        urls=[
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz',
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz',
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz'
        ]
    )

    print("Building German Vocabulary ...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index="de"),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index="en"),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    # Default index to return when OOV (out of vocab)
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt



def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    # Load the dataset
    train_iter, valid_iter, test_iter = multi30k_dataset(
        train=True,
        dev=True,
        test=True,
        urls=[
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz',
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz',
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz'
        ]
    )

    train_iter_map = to_map_style_dataset(train_iter)
    valid_iter_map = to_map_style_dataset(valid_iter)

    # Create dataloaders to iterate over datasets
    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader

def train_worker(
    gpu,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    args
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = create_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=args.batch_size,
        max_padding=args.max_padding
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=args.warmup
        ),
    )
    train_state = TrainState()

    for epoch in range(args.num_epochs):
        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
            accum_iter=args.accum_iter,
            train_state=train_state,
        )

        GPUtil.showUtilization()
        
        # Save model
        file_path = "%s%.2d.pt" % (args.file_prefix, epoch)
        torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    # Save final model checkpoint
    file_path = "%sfinal.pt" % args.file_prefix
    torch.save(module.state_dict(), file_path)


def load_trained_model(args):
    
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_worker(0, vocab_src, vocab_tgt, spacy_de, spacy_en, args)

    model = create_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model


## Eval

def eval_model(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    beam_search,
    beam_size,
    verbose=True,
    max_iter=None,
    pad_idx=2,
    eos_string="</s>",
):
    if max_iter is None:
        results = [()] * len(valid_dataloader)
    else:
        results = [()] * max_iter
        
    if verbose:
        print("Number of validation examples: ", len(valid_dataloader))
        
    if beam_search:
        out_file = open('out_beam.txt', 'w')
    else:
        out_file = open('out_greedy.txt', 'w')
    
    for idx, b in enumerate(valid_dataloader):
        if verbose:
            print("\nExample %d ========\n" % idx)
            
        if max_iter is not None and idx >= max_iter:
            break

        rb = Batch(b[0], b[1], pad_idx)

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        if verbose:
            print(
                "Source Text (Input)        : "
                + " ".join(src_tokens).replace("\n", "")
            )
            print(
                "Target Text (Ground Truth) : "
                + " ".join(tgt_tokens).replace("\n", "")
            )
        
        if beam_search:
            model_out = beam_search_decode(model, rb.src, rb.src_mask, 72, 0, beam_size, vocab_tgt.get_stoi()["</s>"])[0]
        else:
            model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        if verbose:
            print("Model Output               : " + model_txt.replace("\n", ""))
            
        # Write output to file
        out_file.write(" ".join(tgt_tokens).replace("\n", "") + "\n")
        out_file.write(model_txt + "\n")
        out_file.write("\n")
        
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
            
    refs = [" ".join(x[2]).replace("\n", "") for x in results]
    hyps = [x[-1] for x in results]
    score = compute_corpus_level_bleu(refs, hyps)
    out_file.close()    
    
    return results, score


def run_model_eval(beam_search=False, beam_size=4, verbose=True, num_ex=None):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        0,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1
    )
    
    if num_ex is not None:
        valid_dataloader = valid_dataloader[:num_ex]

    print("Loading Trained Model ...")

    model = create_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )
    model.cuda(0)
    model.eval()

    print("Checking Model Outputs:")
    example_data, score = eval_model(
        valid_dataloader, model, vocab_src, vocab_tgt, beam_search, beam_size, verbose
    )
    return model, example_data, score



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use for training")
    parser.add_argument("--num_epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--accum_iter", type=int, default=10, help="Number of gradient accumulation steps")
    parser.add_argument("--base_lr", type=float, default=1.0, help="Base learning rate")
    parser.add_argument("--max_padding", type=int, default=72, help="Maximum sequence length")
    parser.add_argument("--warmup", type=int, default=3000, help="Number of warmup steps")
    parser.add_argument("--file_prefix", type=str, default="multi30k_model_", help="file prefix to use for saving")
    parser.add_argument("--beam_search", action="store_true", help="Use beam search decoding instead of greedy decoding")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size during beam search decoding")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    load_trained_model(args)
    
    _, results, score = run_model_eval(args.beam_search, args.beam_size, verbose=True)
    
    print("Bleu Score: ", score)
    
    