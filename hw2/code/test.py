import os
import sys
import argparse
import torch

def test_attention():
    
    from layers import attention
    
    d_k = 10
    d_v = 15
    key = torch.rand(3, 10, d_k)
    query = torch.rand(3, 10, d_k)
    value = torch.rand(3, 10, d_v)
    out, attn = attention(query, key, value)
    
    assert out.shape == value.shape, print("Incorrect shape of output")
    print("=" * 10 + "   Attention Unit Test 1 Passed   " + "="*10)
    
    sum_attn = torch.sum(attn, dim=-1)
    assert torch.allclose(sum_attn, torch.ones_like(sum_attn).type_as(sum_attn)), print("Incorrect attention weights", sum_attn)
    print("=" * 10 + "   Attention Unit Test 2 Passed   " + "="*10)
    
    mask = torch.ones(3, 10)
    mask[:, -1] = 0
    out, attn = attention(query, key, value, mask=mask)
    assert torch.allclose(attn[:, :, -1], torch.zeros_like(attn[:, :, -1])), print("Attention weights are incorrectly masked")
    print("=" * 10 + "   Attention Unit Test 3 Passed   " + "="*10)
    

def test_multiheaded_attention():
    
    from layers import MultiHeadedAttention
    
    d_k = 16
    d_v = 16
    h = 4
    key = torch.rand(3, 10, d_k)
    query = torch.rand(3, 10, d_k)
    value = torch.rand(3, 10, d_v)
    mattn = MultiHeadedAttention(h, d_k, dropout=0)
    out = mattn(query, key, value)
    
    assert out.shape == value.shape, print("Incorrect shape of output")
    print("=" * 10 + "   Multiheaded Attention Unit Test 1 Passed   " + "="*10)
    
    assert mattn.attn is not None, print("You should set self.attn values in the class")
    attn_size = torch.numel(mattn.attn)
    assert attn_size == 1200, print("Incorrect number of heads / size of attn values")
    print("=" * 10 + "   Multiheaded Attention Unit Test 2 Passed   " + "="*10)
    
    sum_attn = torch.sum(mattn.attn, dim=-1)
    assert torch.allclose(sum_attn, torch.ones_like(sum_attn).type_as(sum_attn)), print("Incorrect attention weights", sum_attn)
    print("=" * 10 + "   Multiheaded Attention Unit Test 3 Passed   " + "="*10)
    

def test_positional_encoding():
    
    from utils import PositionalEncoding
    position = PositionalEncoding(d_model=10, dropout=0.5)
    pos_1 = torch.tensor([8.4147e-01,  5.4030e-01,  1.5783e-01,  9.8747e-01,  2.5116e-02,
           9.9968e-01,  3.9811e-03,  9.9999e-01,  6.3096e-04,  1.0000e+00])
    pos_2 = torch.tensor([ 9.0930e-01, -4.1615e-01,  3.1170e-01,  9.5018e-01,  5.0217e-02,
           9.9874e-01,  7.9621e-03,  9.9997e-01,  1.2619e-03,  1.0000e+00])
        
    assert torch.allclose(pos_1, position.pe[1, :], atol=1e-02, rtol=1e-02) and torch.allclose(pos_2, position.pe[2, :], atol=1e-02, rtol=1e-02), print("Incorrect values in positional embeddings")
    print("=" * 10 + "   Positional Encoding Unit Test 1 Passed   " + "="*10)
    
    
def test_beam_search():
    
    from main import eval_model, load_tokenizers, load_vocab, create_dataloaders, create_model
    
    #global vocab_src, vocab_tgt, spacy_de, spacy_en
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    
    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        0,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
    )
    
    #valid_dataloader = valid_dataloader[:5]

    model = create_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )
    model.cuda(0)
    model.eval()

    results_greedy, score = eval_model(
        valid_dataloader, model, vocab_src, vocab_tgt, beam_search=False, beam_size=1, verbose=False, max_iter=5
    )
        
    
    results_beam, score = eval_model(
        valid_dataloader, model, vocab_src, vocab_tgt, beam_search=True, beam_size=1, verbose=False, max_iter=5
    )
    
    match = True
    for k in range(5):
        if results_greedy[k][-1] != results_beam[k][-1]:
            match = False
            break
    
    assert match, print("Beam search output with beam_size 1 should match greedy decoding")
    print("=" * 10 + "   Beam Search Unit Test 1 Passed   " + "="*10)   
    
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--attention", action="store_true", help="unit tests for attention")
    parser.add_argument("--multiheaded_attention", action="store_true", help="unit test for mulitheaded attn")
    parser.add_argument("--positional_encoding", action="store_true", help="unit test for positional encoding")
    parser.add_argument("--beam_search", action="store_true", help="unit tests for beam search")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if args.attention:
        test_attention()
        
    if args.multiheaded_attention:
        test_multiheaded_attention()
        
    if args.positional_encoding:
        test_positional_encoding()
        
    if args.beam_search:
        test_beam_search()
    