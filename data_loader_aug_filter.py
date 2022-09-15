import random
import numpy as np


class DataLoader(object):
    def __init__(self, dataset, parameter, step="train"):
        self.curr_rel_idx = 0
        self.tasks = dataset[step + "_tasks"]
        self.rel2candidates = dataset["rel2candidates"]
        self.e1rel_e2 = dataset["e1rel_e2"]
        self.all_rels = sorted(list(self.tasks.keys()))
        self.num_rels = len(self.all_rels)
        self.few = parameter["few"]
        self.max_num = parameter["aug_max_num"]
        self.bs = parameter["batch_size"]
        self.nq = parameter["num_query"]
        self.task_aug = dataset["task_aug_dic"]

        if step != "train":
            self.eval_triples = []
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few :])
            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0

    def get_aug_support(self,support_triples,rel,max_num):
      if rel in self.task_aug.keys():
        aug = []
        for i in range(3):
            if len(self.task_aug[rel][i+1]) > 0:
                aug.append(self.task_aug[rel][i+1])
            diminished = [token for st in aug for token in st]
            if 0 < len(diminished) <= max_num:
                for item in diminished:
                  support_triples.append(item)
            elif len(diminished) > max_num:
                top_rel = min(len(self.task_aug[rel][i+1]),max_num)
                for j in range(top_rel):
                    support_triples.append(diminished[j])
                if max_num > top_rel:
                    num = min(max_num - top_rel,len(diminished))
                    for i in range(num):
                        support_triples.append(random.choice(diminished))

        return support_triples

    def next_one(self):
        # shift curr_rel_idx to 0 after one circle of all relations
        if self.curr_rel_idx % self.num_rels == 0:
            random.shuffle(self.all_rels)
            self.curr_rel_idx = 0

        # get current relation and current candidates
        curr_rel = self.all_rels[self.curr_rel_idx]  # str
        self.curr_rel_idx = (
            self.curr_rel_idx + 1
        ) % self.num_rels  # shift current relation idx to next
        curr_cand = self.rel2candidates[curr_rel]
        while (
            len(curr_cand) <= 10 or len(self.tasks[curr_rel]) <= 10
        ):  # ignore the small task sets
            curr_rel = self.all_rels[self.curr_rel_idx]
            self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
            curr_cand = self.rel2candidates[curr_rel]

        # get current tasks by curr_rel from all tasks and shuffle it
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        # 选了few+num query个task
        curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few + self.nq)
        # support集大小为 few
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[: self.few]]
        support_triples = self.get_aug_support(support_triples, curr_rel,self.max_num)
        # query集大小为 num_query
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few :]]

        # construct support and query negative triples
        support_negative_triples = []
        for triple in support_triples:
            e1, rel, e2 = triple
            # for i in range(self.nsn):
            while True:
                negative = random.choice(curr_cand)
                if e1 + rel not in self.e1rel_e2.keys():
                    break
                elif (negative not in self.e1rel_e2[e1 + rel]) and negative != e2:
                    break
            support_negative_triples.append([e1, rel, negative])

        negative_triples = []
        for triple in query_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand)
                if (negative not in self.e1rel_e2[e1 + rel]) and negative != e2:
                    break
            negative_triples.append([e1, rel, negative])

        # support_triples = self.get_aug_support(support_triples,curr_rel)

        return (
            support_triples,
            support_negative_triples,
            query_triples,
            negative_triples,
            curr_rel,
        )


    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.bs)]

        support, support_negative, query, negative, curr_rel = zip(
            *next_batch_all
        )  # 加*号的是解封装
        # print(len(curr_rel))
        # for r in curr_rel:
        #     if len(r)==1:
        #         print(r)
        return [support, support_negative, query, negative], curr_rel

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        # get current triple
        query_triple = self.eval_triples[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[: self.few]
        support_triples = self.get_aug_support(support_triples, curr_rel,self.max_num)

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            # for i in range(self.nsn):
            while True:
                # if shift == len(curr_cand):
                #     negative = e1
                #     break
                negative = curr_cand[shift]
                if e1 + rel not in self.e1rel_e2.keys():
                    break
                elif (negative not in self.e1rel_e2[e1 + rel]) and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) and negative != e2:
                negative_triples.append([e1, rel, negative])
        if len(negative_triples) == 0:
            negative_triples.append([e1, rel, e1])
        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return (
            [support_triples, support_negative_triples, query_triple, negative_triples],
            curr_rel,
        )

    def next_one_on_eval_by_relation(self, curr_rel):
        if self.curr_tri_idx == len(self.tasks[curr_rel][self.few :]):
            self.curr_tri_idx = 0
            return "EOT", "EOT"

        # get current triple
        query_triple = self.tasks[curr_rel][self.few :][self.curr_tri_idx]
        self.curr_tri_idx += 1
        # curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[: self.few]
        support_triples = self.get_aug_support(support_triples, curr_rel, self.max_num)

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if e1 + rel not in self.e1rel_e2.keys():
                    break
                elif (negative not in self.e1rel_e2[e1 + rel]) and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) and negative != e2:
                negative_triples.append([e1, rel, negative])

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return (
            [support_triples, support_negative_triples, query_triple, negative_triples],
            curr_rel,
        )
