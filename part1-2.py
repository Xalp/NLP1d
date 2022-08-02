import numpy as np
# Transition probability
def transition(train):
  with open(train,"r") as f:
    data = f.read().rstrip().splitlines()

  count_bigram = {}
  count_unigram = {}

  count_unigram["START"] = 0
  count_unigram["STOP"] = 0

  for i in range(len(data)):
    inst = data[i]
    # if this is an empty line:
    if len(inst) == 0:
      # building with previous inst
      tag = "STOP"
      if i != 0:
        prev_inst = data[i-1]
        prev_tag = prev_inst.split()[-1]
        if (tag, prev_tag) not in count_bigram.keys():
          count_bigram[(tag, prev_tag)] = 1
        else:
          count_bigram[(tag, prev_tag)] += 1
      # building with the next inst
      tag = "START"
      if i != len(data) - 1:
        next_inst = data[i+1]
        next_tag = next_inst.split()[-1]
        if (tag, next_tag) not in count_bigram.keys():
          count_bigram[(tag, next_tag)] = 1
        else:
          count_bigram[(tag, next_tag)] += 1
      # increase the count of start and stop
      count_unigram["START"] += 1
      count_unigram["STOP"] += 1
      continue
    # do in all cases:
    tag = inst.split()[-1]
    if tag not in count_unigram.keys():
      count_unigram[tag] = 1
    else:
      count_unigram[tag] += 1
    # if not the last inst:
    if i != len(data) - 1:
      next_inst = data[i+1]
      if len(next_inst) == 0:
        continue
      next_tag = next_inst.split()[-1]
      if (tag, next_tag) not in count_bigram.keys():
        count_bigram[(tag, next_tag)] = 1
      else:
        count_bigram[(tag, next_tag)] += 1

  # calculate transition matrix
  f = {}
  q = {}
  for (y_1, y_2), count_bi in count_bigram.items():
    _str_ = "transition:" + y_1 + "+" + y_2
    q[(y_1, y_2)] = count_bi / count_unigram[y_1]
    if count_bi == 0:
      f[_str_] = - np.inf
    else:
      f[_str_] = np.log(count_bi / count_unigram[y_1])
  return q, f

q,f = transition("/content/NLP1d/dataset/train")
print(q)
print(f)