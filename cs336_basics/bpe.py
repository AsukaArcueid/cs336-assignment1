import regex as re
import os
from typing import BinaryIO
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import time
from collections.abc import Iterable, Iterator
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(input_path, start, end, special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end-start).decode('utf-8')
        
    sorted_special = sorted(special_tokens, key=len, reverse=True)
    special_pattern = "|".join(re.escape(st) for st in sorted_special)
    special_matches = list(re.finditer(special_pattern, chunk))
    parts = []
    last_pos = 0
    for m in special_matches:
        parts.append(chunk[last_pos:m.start()])
        last_pos = m.end()
    parts.append(chunk[last_pos:])
    
    total_count = Counter()
    
    for part in parts:
        if not part: continue
        for match in re.finditer(PAT, part):
            token_bytes = tuple(match.group().encode("utf-8"))
            total_count[token_bytes] += 1
            
    return total_count

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096 
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position) 
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

def merge(total_count, counts_of_pairs, pair_to_words, merges, next_id, vocab):
    if not counts_of_pairs:
        return total_count, counts_of_pairs, pair_to_words, merges, vocab

    max_freq = max(counts_of_pairs.values())
    candidates = [p for p, freq in counts_of_pairs.items() if freq == max_freq]
    
    # 平局消除逻辑 (Tie-breaking)
    # 当有多个频率相同的 pair 时，按照 vocab 中对应的字节串大小来选择最大的
    if len(candidates) == 1:
        best_pair = candidates[0]
    else:
        best_pair = max(
            candidates,
            key=lambda p: (vocab[p[0]], vocab[p[1]])
        )

    # 记录字典和合并历史
    merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
    vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

    # 倒排索引
    affected_words = list(pair_to_words.get(best_pair, []))

    for word in affected_words:
        count = total_count.pop(word)

        # 剥离旧词的配对贡献
        for i in range(len(word) - 1):
            old_pair = (word[i], word[i+1])
            counts_of_pairs[old_pair] -= count
            if counts_of_pairs[old_pair] <= 0:
                del counts_of_pairs[old_pair]
            
            pair_to_words[old_pair].discard(word)

        # 生成新词
        new_word_list = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                new_word_list.append(next_id)
                i += 2
            else:
                new_word_list.append(word[i])
                i += 1
        new_word = tuple(new_word_list)

        # 注入新词的配对贡献
        total_count[new_word] += count
        for i in range(len(new_word) - 1):
            new_pair = (new_word[i], new_word[i+1])
            counts_of_pairs[new_pair] += count
            pair_to_words[new_pair].add(new_word)

    return total_count, counts_of_pairs, pair_to_words, merges, vocab

def tokenize(input_path: str, vocab_size: int, special_tokens: list[str], num_workers: int = 6):
    with open(input_path, "rb") as f:
        merges = []
        
        # 初始化阶段不再添加 special_tokens，让 base byte ID 保持为 0-255
        vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256 
            
        t0 = time.perf_counter()  # 计时（chunk）

        total_count = Counter()
        boundaries = find_chunk_boundaries(f, num_workers, special_tokens[0].encode('utf-8'))
        args_list = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        t1 = time.perf_counter()  # 计时（pre-tokenization）

        print("Starting multiprocessing pre-tokenization...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(pre_tokenize, *zip(*args_list)))
        for r in results:
            total_count.update(r)
            
        t2 = time.perf_counter() # 计时（初始化pair及索引）

        merge_time = vocab_size - 256 - len(special_tokens)
        counts_of_pairs = Counter()
        pair_to_words = defaultdict(set)
        
        print("Building initial pairs and inverted index...")
        for blist, freq in total_count.items():
            for i in range(len(blist) - 1):
                pair = (blist[i], blist[i+1])
                counts_of_pairs[pair] += freq
                pair_to_words[pair].add(blist)

        t3 = time.perf_counter() # 计时（merge）

        print(f"Starting {merge_time} merges...")
        for step in range(merge_time):
            total_count, counts_of_pairs, pair_to_words, merges, vocab = merge(
                total_count, counts_of_pairs, pair_to_words, merges, next_id, vocab
            )
            next_id += 1
            
            if (step + 1) % 50 == 0 or (step + 1) == merge_time:
                print(f"Merged {step + 1}/{merge_time} pairs. Current Temp Vocab Size: {len(vocab)}")
                
        # 所有 merge 结束后，执行 ID 整体平移，将 special_tokens 放在 0 开头
        offset = len(special_tokens)
        new_vocab = {}
        
        # 1. 优先放入特殊字符 (ID 从 0 开始)
        for i, st in enumerate(special_tokens):
            new_vocab[i] = st.encode("utf-8", errors="ignore")
            
        # 2. 将原有的 base bytes 和合并出来的新 token 整体往后推 offset 个单位
        for old_id, b in vocab.items():
            new_vocab[old_id + offset] = b
            
        vocab = new_vocab

        t4 = time.perf_counter() # 计时结束
        print(f"find_boundaries: {t1-t0:.2f}s")
        print(f"pre_tokenize:    {t2-t1:.2f}s")
        print(f"build pairs:     {t3-t2:.2f}s")
        print(f"merges:          {t4-t3:.2f}s")
        
    return vocab, merges

def save(vocab, merges, vocab_path, merges_path):
    json_vocab = {str(k): list(v) for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(json_vocab, f)
    
    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = ",".join(map(str, list(p1)))
            s2 = ",".join(map(str, list(p2)))
            f.write(f"{s1} {s2}\n")


###-----------------------------------------------------分割线--------------------------------------------------###

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        if special_tokens is not None:
            vocab_bytes = set(self.vocab.values())
            unique_st = []
            for st in special_tokens:
                b = st.encode('utf-8')
                if b not in vocab_bytes:
                    unique_st.append(b)

            stlen = len(unique_st)
            vocab_len=len(self.vocab)
            for i in range(stlen):
                vocab[i+vocab_len]=unique_st[i]
        self.offset = next(k for k, v in self.vocab.items() if v == b'\x00')
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_map = {}
        for i, (p0, p1) in enumerate(merges):
            id0 = self.byte_to_id[p0]
            id1 = self.byte_to_id[p1]
            # 拼接 bytes，去 byte_to_id 查它真实的 ID
            combined_bytes = p0 + p1
            self.merge_map[(id0, id1)] = self.byte_to_id[combined_bytes]
        self.byte_to_id_map_origin = {}
        for i in range(256):
            byte_obj = bytes([i])
            # 在你已经建立好的 self.byte_to_id (bytes -> id) 中查找
            if byte_obj in self.byte_to_id:
                self.byte_to_id_map_origin[i] = self.byte_to_id[byte_obj]
            else:
                # 如果词表不完整，至少报错或记录，这通常意味着词表有问题
                pass



    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding = 'utf-8')as f:
            raw_vocab = json.load(f)
            vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}

            
        merges = []
        with open(merges_filepath, "r", encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                if len(parts) == 2:
                    p0 = bytes(map(int, parts[0].split(',')))
                    p1 = bytes(map(int, parts[1].split(',')))
                    merges.append((p0, p1)) 
        
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_pat = "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
            parts = re.split(special_pat, text)
        else:
            parts=[text]
        
        words = []
        for part in parts:
            if not part: continue
            if part in self.special_tokens:
                words.append(part)
            else:
                matches = re.finditer(PAT,part)
                for m in matches:
                    words.append(m.group())
        
        result=[]
        for word in words:
            word_bytes=word.encode('utf-8')
            if word_bytes in self.byte_to_id:
                result.append(self.byte_to_id[word_bytes])
            else:
                result+=self.encode_word(word)
        return result
    
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    
    def decode(self, ids: list[int]) -> str:
        ans=b''
        for i in ids:
            ans+=self.vocab.get(i)
        return ans.decode('utf-8',errors='replace')

    def encode_word(self,word:str)->list[int]:
        word_bytes = word.encode('utf-8')
        word = [self.byte_to_id_map_origin[b] for b in word_bytes]


        while True:
            best_rank = float('inf')
            best_idx = -1
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                if pair in self.merge_map:
                    rank=self.merge_map[pair]
                    if rank<best_rank:
                        best_rank,best_idx=rank,i
            if best_idx==-1:
                return word
            else:
                new_word=[]
                i=0
                while i < len(word):
                    if i == best_idx:
                        new_word.append(best_rank)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word=new_word















if __name__ == "__main__":
    '''vocab, merges = tokenize('/home/zhang/projects/cs336/cs336-assignment1/data/TinyStoriesV2-GPT4-train.txt', 50000, ['<|endoftext|>'], 10)
    #print(vocab)
    print('\n')
    longest = max(vocab.items(), key=lambda x: len(x[1]))
    print(f"Longest token: {longest[1]} (id={longest[0]}, len={len(longest[1])})")
    save(vocab,merges,'/home/zhang/projects/cs336/cs336-assignment1/bpe_tinystories/vocab.json','/home/zhang/projects/cs336/cs336-assignment1/bpe_tinystories/merge.txt')'''
    tokenizer=Tokenizer.from_files('/home/zhang/projects/cs336/cs336-assignment1/bpe_tinystories/vocab.json','/home/zhang/projects/cs336/cs336-assignment1/bpe_tinystories/merge.txt',['<|im_start|>','<|endoftext|>'])
    encoded =tokenizer.encode('Hello world! <|endoftext|> This is a BPE tokenizer test with 12345 numbers and "multiple" symbols. <|im_start|>')
    print(encoded)
    print(tokenizer.decode(encoded))