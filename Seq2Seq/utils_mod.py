import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys
from torchtext.data.metrics import bleu_score
from collections import Counter
import math

spacy_ger = spacy.load("de_core_news_sm")
def tokenize_ger(text):
    return [tok.text.lower() for tok in spacy_ger.tokenizer(text)]

def translate_sentence(model, sentence, german, english, device, beam_width, max_length=50):
    model.eval()
    
    if isinstance(sentence, str):
        tokens = tokenize_ger(sentence)
    else:
        tokens = sentence

    # Add <sos> and <eos> tokens
    tokens = [german.init_token] + tokens + [german.eos_token]
    src_indexes = [german.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    # Encode the source sentence
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    # Initialize beam search
    beams = [(0, [english.vocab.stoi[english.init_token]], hidden, cell)]
    
    for _ in range(max_length):
        new_beams = []
        for score, seq, dec_hidden, dec_cell in beams:
            if seq[-1] == english.vocab.stoi[english.eos_token]:
                new_beams.append((score, seq, dec_hidden, dec_cell))
            else:
                # Decode
                trg_tensor = torch.LongTensor([seq[-1]]).to(device)
                with torch.no_grad():
                    output, dec_hidden, dec_cell = model.decoder(trg_tensor, dec_hidden, dec_cell)
                
                # Get top k predictions
                top_preds = torch.topk(output, k=beam_width, dim=1)
                for i in range(beam_width):
                    new_score = score + top_preds.values[0][i].item()
                    new_seq = seq + [top_preds.indices[0][i].item()]
                    new_beams.append((new_score, new_seq, dec_hidden, dec_cell))
        
        # Keep only the top 'beam_width' beams
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
        
        # Stop if all beams end with <eos>
        if all(b[1][-1] == english.vocab.stoi[english.eos_token] for b in beams):
            break
    
    # Select the best beam
    best_seq = beams[0][1]
    
    # Convert indices to words
    translated_sentence = [english.vocab.itos[idx] for idx in best_seq]
    
    # Remove <sos> and <eos> tokens
    return translated_sentence[1:-1]

def bleu(data, model, german, english, device, beam_width, max_length):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device, beam_width, max_length)

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def modified_bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
    clipped_counts = Counter()
    total_counts = Counter()
    weights = weights[:max_n]
    
    candidate_len = 0
    ref_len = 0

    for (candidate, references) in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)
        ref_len += min(len(ref) for ref in references)

        reference_counters = [Counter(ngrams(ref, n)) for ref in references for n in range(1, max_n + 1)]

        for n in range(1, max_n + 1):
            candidate_counter = Counter(ngrams(candidate, n))
            
            for ngram, count in candidate_counter.items():
                total_counts[n] += count
                clipped_counts[n] += min(count, max(counter.get(ngram, 0) for counter in reference_counters))

    if candidate_len == 0:
        return 0

    bp = 1.0
    if candidate_len < ref_len:
        bp = math.exp(1 - ref_len / candidate_len)

    prec_scores = [clipped_counts[n] / total_counts[n] if total_counts[n] > 0 else 0 for n in range(1, max_n + 1)]
    
    score = bp * math.exp(sum(w * math.log(p) if p > 0 else float('-inf') for w, p in zip(weights, prec_scores)))
    return score

def ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])

def save_checkpoint(state, filename="my_checkpoint_mod.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])