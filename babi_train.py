import numpy as np
import os
import pickle
import model
import random
import babi_graph_parse
from graceful_interrupt import GracefulInterruptHandler
from pprint import pformat

BATCH_SIZE = 10

def convert_answer(answer, num_words, format_spec, maxlen):
    """
    Convert an answer into an appropriate answer matrix given
    a ModelOutputFormat.

    num_words should be after processing with get_effective_answer_words,
    so that the last word is the "stop" word
    """
    assert format_spec in model.ModelOutputFormat
    if format_spec == model.ModelOutputFormat.subset:
        ans_mat = np.zeros((1,num_words), np.float32)
        for word in answer:
            ans_mat[0, word] = 1.0
    elif format_spec == model.ModelOutputFormat.category:
        ans_mat = np.zeros((1,num_words), np.float32)
        ans_mat[0,answer[0]] = 1.0
    elif format_spec == model.ModelOutputFormat.sequence:
        ans_mat = np.zeros((maxlen+1,num_words), np.float32)
        for i,word in enumerate(answer+[num_words-1]*(maxlen+1-len(answer))):
            ans_mat[i, word] = 1.0
    return ans_mat

def get_effective_answer_words(answer_words, format_spec):
    """
    If needed, modify answer_words using format spec to add padding chars
    """
    if format_spec == model.ModelOutputFormat.sequence:
        return answer_words + ["<stop>"]
    else:
        return answer_words

def sample_batch(matching_stories, batch_size, num_answer_words, format_spec):
    chosen_stories = [random.choice(matching_stories) for _ in range(batch_size)]
    return assemble_batch(chosen_stories, num_answer_words, format_spec)

def assemble_batch(stories, num_answer_words, format_spec):
    sents, graphs, queries, answers = zip(*stories)
    cvtd_sents = np.array(sents, np.int32)
    cvtd_queries = np.array(queries, np.int32)
    max_ans_len = max(len(a) for a in answers)
    cvtd_answers = np.stack([convert_answer(answer, num_answer_words, format_spec, max_ans_len) for answer in answers])
    num_new_nodes, new_node_strengths, new_node_ids, next_edges = zip(*graphs)
    num_new_nodes = np.stack(num_new_nodes)
    new_node_strengths = np.stack(new_node_strengths)
    new_node_ids = np.stack(new_node_ids)
    next_edges = np.stack(next_edges)
    return cvtd_sents, cvtd_queries, cvtd_answers, num_new_nodes, new_node_strengths, new_node_ids, next_edges

def visualize(m, story_buckets, wordlist, answerlist, output_format, outputdir, batch_size=1, seq_len=5, debugmode=False):
    cur_bucket = random.choice(story_buckets)
    sampled_batch = sample_batch(cur_bucket, batch_size, len(answerlist), output_format)
    part_sampled_batch = sampled_batch[:3]
    with open(os.path.join(outputdir,'stories.txt'),'w') as f:
        babi_graph_parse.print_batch(part_sampled_batch, wordlist, answerlist, file=f)
    with open(os.path.join(outputdir,'answer_list.txt'),'w') as f:
        f.write('\n'.join(answerlist) + '\n')
    if debugmode:
        args = sampled_batch
        fn = m.debug_test_fn
        print("FALKHVKADHL")
    else:
        args = part_sampled_batch[:2] + ((seq_len,) if output_format == model.ModelOutputFormat.sequence else ())
        fn = m.test_fn
    results = fn(*args)
    for i,result in enumerate(results):
        np.save(os.path.join(outputdir,'result_{}.npy'.format(i)), result)

def train(m, story_buckets, len_answers, output_format, num_updates, outputdir, start=0, batch_size=BATCH_SIZE, validation_buckets=None):
    with GracefulInterruptHandler() as interrupt_h:
        for i in range(start+1,start+num_updates+1):
            cur_bucket = random.choice(story_buckets)
            sampled_batch = sample_batch(cur_bucket, batch_size, len_answers, output_format)
            loss, info = m.train(*sampled_batch)
            if np.any(np.isnan(loss)):
                print("Loss at timestep {} was nan! Aborting".format(i))
                break
            with open(os.path.join(outputdir,'data.csv'),'a') as f:
                if i == 1:
                    f.seek(0)
                    f.truncate()
                    keylist = "iter, loss, " + ", ".join(k for k,v in sorted(info.items())) + "\n"
                    f.write(keylist)
                    if validation_buckets is not None:
                        with open(os.path.join(outputdir,'valid.csv'),'w') as f2:
                            f2.write(keylist)
                f.write("{}, {},".format(i,loss) + ", ".join(str(v) for k,v in sorted(info.items())) + "\n")
            if i % 1 == 0:
                print("update {}: {}\n{}".format(i,loss,pformat(info)))
            if i % 1000 == 0:
                if validation_buckets is not None:
                    cur_bucket = random.choice(validation_buckets)
                    sampled_batch = sample_batch(cur_bucket, batch_size, len_answers, output_format)
                    valid_loss, valid_info = m.eval(*sampled_batch)
                    print("validation at {}: {}\n{}".format(i,valid_loss,pformat(valid_info)))
                    with open(os.path.join(outputdir,'valid.csv'),'a') as f:
                        f.write("{}, {}, ".format(i,valid_loss) + ", ".join(str(v) for k,v in sorted(valid_info.items())) + "\n")
                pickle.dump(m.params, open(os.path.join(outputdir, 'params{}.p'.format(i)), 'wb'))
            if interrupt_h.interrupted:
                break
