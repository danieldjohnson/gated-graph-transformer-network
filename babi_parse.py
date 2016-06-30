import os
import re
import collections
import numpy as np

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return re.findall('(?:\w+)|\S',sent)


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            a = [x.strip() for x in a.split(',')]
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = list(map(int, supporting.split()))
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(taskname):
    with open(taskname, 'r') as f:
        lines = f.readlines()
    return parse_stories(lines)

def get_max_sentence_length(stories):
    return max((max(max((len(sentence) for sentence in story[0])), len(story[1])) for story in stories))

def get_buckets(stories, max_ignore_unbatched=100, max_pad_amount=25):
    sentencecounts = [len(story[0]) for story in stories]
    countpairs = sorted(collections.Counter(sentencecounts).items())

    buckets = []
    smallest_left_val = 0
    num_unbatched = max_ignore_unbatched
    for val,ct in countpairs:
        num_unbatched += ct
        if val - smallest_left_val > max_pad_amount or num_unbatched > max_ignore_unbatched:
            buckets.append(val)
            smallest_left_val = val
            num_unbatched = 0
    if buckets[-1] != countpairs[-1][0]:
        buckets.append(countpairs[-1][0])

    return buckets

PAD_WORD = "<PAD>"

def get_wordlist(stories):
    words = sorted([PAD_WORD] + list(set((word
        for story in stories
        for sentence in (story[0] + [story[1]])
        for word in sentence ))))
    wordmap = {word:i for i,word in enumerate(words)}
    return words, wordmap

def get_answer_list(stories):
    words = sorted(list(set(word for s,q,answer in stories for word in answer)))
    wordmap = {word:i for i,word in enumerate(words)}
    return words, wordmap

def pad_story(story, num_sentences, sentence_length):
    def pad(lst,dlen,pad):
        return lst + [pad]*(dlen - len(lst))
    
    return ([pad(s,sentence_length,PAD_WORD) for s in pad(story[0],num_sentences,[])],
            pad(story[1],sentence_length,PAD_WORD),
            story[2])

def convert_story(story, wordmap, answer_map):
    return ([[wordmap[w] for w in s] for s in story[0]],
            [wordmap[w] for w in story[1]],
            [answer_map[w] for w in story[2]])

def bucket_stories(stories, buckets, wordmap, answer_map, sentence_length):
    def process_story(s,bucket_len):
        return convert_story(pad_story(s, bucket_len, sentence_length), wordmap, answer_map)
    return [ [process_story(story,bmax) for story in stories if bstart < len(story[0]) <= bmax]
                for bstart, bmax in zip([0]+buckets,buckets)]

def prepare_stories(stories):
    sentence_length = get_max_sentence_length(stories)
    buckets = get_buckets(stories)
    wordlist, wordmap = get_wordlist(stories)
    anslist, ansmap = get_answer_list(stories)
    bucketed = bucket_stories(stories, buckets, wordmap, ansmap, sentence_length)
    return sentence_length, buckets, wordlist, anslist, bucketed

def print_batch(story, wordlist, anslist):
    sents, query, answer = story
    for batch,(s,q,a) in enumerate(zip(sents,query,answer)):
        print("Story {}".format(batch))
        for sent in s:
            print(" ".join([wordlist[word] for word in sent]))
        print(" ".join(wordlist[word] for word in q))
        print(" ".join(anslist[word] for word in a))
