import numpy as np
import model
import random
import babi_parse
from graceful_interrupt import GracefulInterruptHandler

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

def assemble_batch(matching_stories, batch_size, num_answer_words, format_spec):
    sents, queries, answers = zip(*(random.choice(matching_stories) for _ in range(batch_size)))
    cvtd_sents = np.array(sents, np.int32)
    cvtd_queries = np.array(queries, np.int32)
    max_ans_len = max(len(a) for a in answers)
    cvtd_answers = np.stack([convert_answer(answer, num_answer_words, format_spec, max_ans_len) for answer in answers])
    return cvtd_sents, cvtd_queries, cvtd_answers

def train(m, story_buckets, len_answers, output_format, num_updates, outputdir, start=0, batch_size=BATCH_SIZE):
    with GracefulInterruptHandler() as interrupt_h:
        for i in range(start+1,start+num_updates+1):
            cur_bucket = random.choice(bucketed)
            sampled_batch = assemble_batch(cur_bucket, batch_size, len_answers, output_format)
            loss = m.train_fn(*sampled_batch)
            with open(os.path.join(outputdir,'data.csv'),'a') as f:
                if i == 1:
                    f.seek(0)
                    f.truncate()
                    f.write("iter, loss, \n")
                f.write("{}, {}".format(i,loss))
            if i % 1 == 0:
                print("update {}: {}".format(i,loss))
            if i % 1000 == 0:
                pickle.dump(m.params, open(os.path.join(outputdir, 'params{}.p'.format(i)), 'wb'))
            if interrupt_h.interrupted:
                break
