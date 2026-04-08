import regex as re
import os
from typing import BinaryIO
from collections import Counter


def pre_tokenize(chunk,special_tokens):
    PLACEHOLDER = "|||RESERVED_ST_SEPARATOR|||"
    temp_text = chunk
    for st in special_tokens:
        temp_text = temp_text.replace(st, PLACEHOLDER)
    parts = temp_text.split(PLACEHOLDER)
    PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_counts = Counter()
    for part in parts:
        if not part:
            continue
        
        for match in re.finditer(PAT, part):
            token_str = match.group()
            token_bytes = tuple(token_str.encode("utf-8"))
            token_counts[token_bytes] += 1
    return token_counts

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def merge(total_count,counts_of_pairs,merges,next_id,vocab):
    new_total = Counter()
    most_common = counts_of_pairs.most_common(1) 
    best_pair = most_common[0][0]        
    merges.append((vocab[best_pair[0]],vocab[best_pair[1]]))
    vocab[next_id]=vocab[best_pair[0]]+vocab[best_pair[1]]
    for token, token_freq in total_count.items():
        if best_pair[0] not in token:
            new_total[token] += token_freq
            continue
        for j in range(len(token) - 1):
            pair = (token[j], token[j+1])
            counts_of_pairs[pair] -= token_freq
        new_token_list = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and token[i] == best_pair[0] and token[i+1] == best_pair[1]:
                new_token_list.append(next_id)
                i += 2
            else:
                new_token_list.append(token[i])
                i += 1
        new_token = tuple(new_token_list)
        for j in range(len(new_token) - 1):
            new_pair = (new_token[j], new_token[j+1])
            counts_of_pairs[new_pair] += token_freq
        new_total[new_token] += token_freq
    counts_of_pairs += Counter()
    total_count.clear()
    total_count.update(new_total)
    return total_count,counts_of_pairs,merges,vocab

def tokenize(input_path:str,vocab_size:int,special_tokens:list[str],num_workers:int):
    with open(input_path, "rb") as f:
        merges=[]
        vocab={}
        for i in range(0,256):
            vocab[i]=bytes([i])
        next_id=256
        for st in special_tokens:
            vocab[next_id] = st.encode('utf-8')   # 特殊 token 的 UTF-8 字节串
            next_id += 1
        total_count=Counter()
        boundaries = find_chunk_boundaries(f, num_workers, special_tokens[0].encode('utf-8'))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            total_count.update(pre_tokenize(chunk,special_tokens))
        merge_time=vocab_size-256-len(special_tokens)
        counts_of_pairs=Counter()
        for blist, freq in total_count.items():
            for i in range(0,len(blist)-1):
                counts_of_pairs.update({(blist[i],blist[i+1]):freq})
        for _ in range(merge_time):
            total_count,counts_of_pairs,merges,vocab=merge(total_count,counts_of_pairs,merges,next_id,vocab)
            next_id+=1
    return vocab,merges

vocab,merges=tokenize('/home/zhang/projects/cs336/cs336-assignment1/data/TinyStoriesV2-GPT4-valid.txt',450,['<|endoftext|>'],4)
print(vocab)
print(merges)