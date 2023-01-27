import torch
import json
from collections import OrderedDict
import time
from tqdm import tqdm
import torch.nn as nn
import ipdb
import json
from retro_pytorch import RETRO, TrainingWrapper
from retro_pytorch.training import top_p
from transformers import AutoTokenizer
from ppl_dataloader import GPT2PPLDataset

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
decoding_method = 'sampling'
# decoding_method = 'greedy'

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(
    max_seq_len = 512,                      # max sequence length
    enc_dim = 896,                           # encoder model dimension
    enc_depth = 3,                           # encoder depth
    dec_dim = 768,                           # decoder model dimensions
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
    heads = 12,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
)

wrapper = TrainingWrapper(
    retro = retro,                                 # path to retro instance
    knn = 2,                                       # knn (2 in paper was sufficient)
    chunk_size = 64,                               # chunk size (64 in paper)
    documents_path = './wikitext103_text_folder',              # path to folder of text
    glob = '**/*.txt',                             # text glob
    chunks_memmap_path = './wikitext103_text_folder/train.chunks.dat',     # path to chunks
    seqs_memmap_path = './wikitext103_text_folder/train.seq.dat',          # path to sequence data
    doc_ids_memmap_path = './wikitext103_text_folder/train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
    max_chunks = 10_000_000,                        # maximum cap to chunks
    max_seqs = 2_000_000,                            # maximum seqs
    knn_extra_neighbors = 100,                     # num extra neighbors to fetch
    max_index_memory_usage = '10G',
    current_memory_available = '50G'
)

# packup the model with dataparallel
# load the model checkpoint
model_path = 'best_model_100000.pt'
parameters = torch.load(model_path)
new_data = OrderedDict()
for key, value in parameters.items():
    key = key.replace('module.', '')
    new_data[key] = value
retro.load_state_dict(new_data)
retro = retro.cuda().eval()

max_ctx_len = 384
# 0.95 nucleus sampling
filter_thres = 0.05 if decoding_method == 'sampling' else 2
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')    #  compatible with the GPT2 vocabulary

# set the maximum sequence length for generation (32 prefix + 128 generation)
wrapper.max_seq_len = 200

# load the ppl test set
args = {
    'tokenizer': 'gpt2',
    'ppl_max_len': 200,
}
data = GPT2PPLDataset(**args)
sampler = torch.utils.data.distributed.DistributedSampler(data)
iter_ = DataLoader(data, batch_size=1, collate_fn=data.collate, sampler=sampler)

for prefix, reference in tqdm(iter_):
    prompt = torch.LongTensor(tokenizer.encode(prefix, add_special_tokens=False)).unsqueeze(0).cuda()
    prefix_len = len(tokenizer.decode(prompt[0]))
    # filter_thres larger than 1, lead to the greedy search
    bt = time.time()
    sampled = wrapper.generate(prompt, filter_fn=top_p, filter_thres = filter_thres, temperature = 1.0) # (1, <2049) terminates early if all <eos>
    time_cost = time.time() - bt
    rest = tokenizer.decode(sampled[0])
    text = rest[prefix_len:]
    collection[seed].append({
        'prefix': prefix, 
        'reference': reference, 
        'text': text, 
        'time_cost': time_cost
    })

for seed in collection:
    result = collection[seed]
    with open(f'raw_files/random_runs/wikitext103_retro_result_{decoding_method}_random_seed_{seed}.json', 'w') as f:
        json.dump(result, f, indent=4)
