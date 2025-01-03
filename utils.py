import sys
import copy
import torch
import random
import numpy as np
import time
from collections import defaultdict
from multiprocessing import Process, Queue
from tqdm import tqdm



# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    
    def random_item_from_training():
        user = np.random.choice(list(user_train.keys()))
        return np.random.choice(user_train[user])
    
    
    def sample():

        user = np.random.randint(1, usernum + 1)
        while user not in user_train or len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        
        #Inject noise into 10% of the sequence tokens
        # num_tokens_to_noise = int(0.01 * maxlen)
        # token_indices = np.random.choice(maxlen, num_tokens_to_noise, replace=False)
        # for idx in token_indices:
        #     seq[idx] = random_item_from_training()
            
        # print(f" seq shape: {seq.shape}, pos shape: {pos.shape}, neg shape: {neg.shape}")
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        
        result_queue.put(zip(*one_batch))

# def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
#     def sample():
#         # Randomly choose a user
#         user = np.random.randint(1, usernum + 1)
        
#         # Ensure the user has more than 1 interaction
#         while user not in user_train or len(user_train[user]) <= 1:
#             user = np.random.randint(1, usernum + 1)

#         # Initialize the arrays for seq, pos, and neg
#         seq = np.zeros([maxlen], dtype=np.int32)
#         pos = np.zeros([maxlen], dtype=np.int32)
#         neg = np.zeros([maxlen], dtype=np.int32)

#         # Get the user's interactions
#         interactions = user_train[user]
        
#         # We start at the last interaction
#         nxt = interactions[-1]
#         idx = maxlen - 1  # Start filling from the last position

#         ts = set(interactions)  # Keep track of items already seen
#         for i in reversed(interactions[:-1]):  # Skip the last interaction, it's already used as `nxt`
#             seq[idx] = i
#             pos[idx] = nxt
#             if nxt != 0:
#                 neg[idx] = random_neq(1, itemnum + 1, ts)  # Generate a random negative sample
#             nxt = i
#             idx -= 1
#             if idx == -1: break  # Stop if we've filled the sequence
        
#         # Now generate the shifted sequences, where the new `seq` and `pos` arrays shift for each window
#         batches = []
#         for start in range(0, maxlen):
#             new_seq = np.roll(seq, start)  # Roll the sequence by the start index
#             new_pos = np.roll(pos, start)  # Roll the pos array similarly
#             new_neg = np.roll(neg, start)  # Roll the neg array similarly
#             batches.append((user, new_seq, new_pos, new_neg))  # Collect the result

#         # Return the batch as a tuple
#         return batches

#     np.random.seed(SEED)
#     while True:
#         one_batch = []
#         for i in range(batch_size):
#             one_batch.extend(sample())  # Add multiple sequence shifts per user
        
#         result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    items=defaultdict(int)
    f = open('./data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
       
        items[i] = 1 
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    
    user_count = 0
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
            
        user_count += 1
        
    print('user_count:', user_count)
    print('item count:', len(items))
    return [user_train, user_valid, user_test, usernum, itemnum]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    sumt = 0
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    NDCG_20, HT_20 = 0.0, 0.0
    NDCG_5, HT_5 = 0.0, 0.0


    users = range(1, usernum + 1)
    for u in tqdm(users):
  
        if u not in train or u not in test or len(train[u]) < 1 or len(test[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        # rated.add(0)
        item_idx = [test[u][0]]
        for _ in (range(args.eval_neg_sample)):
            t = np.random.randint(1, itemnum + 1)  #
            while t in rated: t = np.random.randint(1, itemnum + 1)  #
            item_idx.append(t)
        t0 = time.time()
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        t1 = time.time()
        sumt += (t1 - t0) * 1000
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1


        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1


        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1


    return NDCG_5 / valid_user, HT_5 / valid_user, NDCG / valid_user, HT / valid_user, NDCG_20 / valid_user, HT_20 / valid_user, sumt



# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    NDCG_20, HT_20 = 0.0, 0.0
    NDCG_5, HT_5 = 0.0, 0.0


    users = range(1, usernum + 1)
    for u in users:
        if u not in train or u not in test or len(train[u]) < 1 or len(test[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        item_idx = [valid[u][0]]

        for _ in range(args.eval_neg_sample):
            t = np.random.randint(1, itemnum + 1)  #
            while t in rated: t = np.random.randint(1, itemnum + 1)  #
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1


        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1


    return NDCG_5 / valid_user, HT_5 / valid_user, NDCG / valid_user, HT / valid_user, NDCG_20 / valid_user, HT_20 / valid_user