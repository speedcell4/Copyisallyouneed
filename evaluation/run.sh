#!/bin/bash

cuda=$1
# file_name=raw_files/wikitext103_gpt2_result_nucleus_sampling.json
# file_name=raw_files/wikitext103_gpt2_result_greedy.json
# file_name=raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling.json
# file_name=raw_files/wikitext103_neurlab_gpt2_result_nucleus_sampling_v2.json
# file_name=raw_files/en_wiki_knnlm_result_nucleus_sampling_full.json
# file_name=raw_files/en_wiki_knnlm_result_greedy_full.json
# file_name=raw_files/lawmt_gpt2_result_greedy_v2.json
# file_name=raw_files/en_wiki_gpt2_result_greedy_v2.json
# file_name=raw_files/en_wiki_gpt2_result_nucleus_sampling_v2.json
# file_name=raw_files/lawmt_knnlm_result_greedy_full.json
# file_name=raw_files/lawmt_knnlm_result_nucleus_sampling_full.json
# file_name=raw_files/lawmt_retro_result_greedy.json
# file_name=raw_files/lawmt_retro_result_sampling.json
# file_name=raw_files/random_runs_en_wiki_testset/en_wiki_copyisallyouneed_result_nucleus_sampling_on_en_wiki_testset_seed_1000.0_1.0.json
# file_name=raw_files/random_runs_en_wiki_testset/en_wiki_copyisallyouneed_result_nucleus_sampling_on_en_wiki_testset_seed_1.0_1.0.json
# file_name=raw_files/random_runs_en_wiki_testset/en_wiki_copyisallyouneed_result_nucleus_sampling_on_en_wiki_testset_seed_100.0_1.0.json
# file_name=raw_files/random_runs/wikitext103_gpt2_result_nucleus_sampling_on_wikitext103_testset_seed_100.0_1.0.json
# file_name=raw_files/wikitext103_gpt2_result_greedy.json
# file_name=raw_files/wikitext103_knnlm_result_greedy_full.json
# file_name=raw_files/random_runs_gpt2_en_wiki/en_wiki_gpt2_result_nucleus_sampling_on_wikitext103_testset_seed_1.0.json
# file_name=raw_files/random_runs_en_wiki_testset/en_wiki_copyisallyouneed_result_greedy_wikitext_index_on_wikitext103_testset.json
# file_name=raw_files/random_runs_en_wiki/en_wiki_knnlm_result_nucleus_sampling_on_wikitext103_index_wikitext103_testset_seed_5.0.json
# file_name=raw_files/random_runs_en_wiki/en_wiki_knnlm_result_nucleus_sampling_on_wikitext103_index_wikitext103_testset_seed_1.0.json
# file_name=raw_files/wikitext103_gpt2_result_greedy.json
# file_name=raw_files/random_runs_retro/en_wiki_retro_result_greedy_en_wiki_index.json
# file_name=raw_files/random_runs_retro/en_wiki_retro_result_sampling_random_seed_1.0_en_wiki_index.json
# file_name=raw_files/random_runs_wikitext103_testset_knnlm/wikitext103_knnlm_result_nucleus_sampling_full_0.118_0.00785.json
# file_name=raw_files/random_runs_wikitext103_testset_knnlm/wikitext103_knnlm_result_greedy_full_0.118_0.00785.json
# file_name=raw_files/random_runs_lawmt/lawmt_knnlm_result_nucleus_sampling_seed_1.0.json
# file_name=raw_files/random_runs_lawmt/lawmt_knnlm_result_greedy_full_0.118_0.00785.json
file_name=raw_files/random_runs_lawmt/en_wiki_knnlm_result_greedy_full_0.118_0.00785.json

# coherence
# CUDA_VISIBLE_DEVICES=$cuda python coherence/test.py --test_path $file_name

# mauve
# python mauve/test.py --test_path $file_name --device $cuda

# diversity
python diversity/test.py --test_path $file_name
