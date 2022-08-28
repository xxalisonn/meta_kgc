import random
import numpy as np
import torch
import json

class DataLoader(object):
    def __init__(self, dataset, parameter, step="train"):
        self.curr_rel_idx = 0
        self.tasks = dataset[step + "_tasks"]
        self.rel2candidates = dataset["rel2candidates"]
        self.e1rel_e2 = dataset["e1rel_e2"]
        self.all_rels = sorted(list(self.tasks.keys()))
        self.num_rels = len(self.all_rels)
        self.few = parameter["few"]
        self.bs = parameter["batch_size"]
        self.nq = parameter["num_query"]
        self.rel_t = dataset["rel2dom_t"]
        self.rel2nn = dataset["rel2nn"]
        self.ent_conc = dataset["ent2dom"]
        self.conc_ents = dataset["dom2ent"]
        self.ent2id = dataset["ent2id"]
        self.id2ent = dataset["id2ent"]
        self.rel2id = dataset["task_rel2id"]
        self.id2rel = dataset["task_id2rel"]

        if step != "train":
            self.eval_triples = []
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few :])
            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0

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
        curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few + self.nq)
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[: self.few]]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few :]]

        # construct support and query negative triples
        support_negative_triples = []
        # support_negative_triples = get_negative_by_cs()
        # 默认取一个负样本
        for triple in support_triples:
            e1, rel, e2 = triple
            # for i in range(self.nsn):
            neg_cand = self.concept_filter_t(e2, rel)
            while True:
                # negative = random.choice(curr_cand)
                negative = random.choice(neg_cand)
                if (negative not in self.e1rel_e2[e1 + rel]) and negative != e2:
                    break

            support_negative_triples.append([e1, rel, negative])

        negative_triples = []
        for triple in query_triples:
            e1, rel, e2 = triple
            neg_cand = self.concept_filter_t(e2, rel)
            while True:
                # negative = random.choice(curr_cand)
                negative = random.choice(neg_cand)
                if (negative not in self.e1rel_e2[e1 + rel]) and negative != e2:
                    break
            negative_triples.append([e1, rel, negative])

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
                # neg_cand = self.concept_filter_t(e2, rel)
                # negative = random.choice(neg_cand)
                if (negative not in self.e1rel_e2[e1 + rel]) and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        neg_cand = self.concept_filter_t(e2, rel)
        # for negative in curr_cand:
        for negative in neg_cand:
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

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            neg_cand = self.concept_filter_t(e2, rel)
            negative = random.choice(neg_cand)
            while True:
                # negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) and negative != e2:
                    break
                else:
                    negative = random.choice(neg_cand)

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
        print('stop here')

        return (
            [support_triples, support_negative_triples, query_triple, negative_triples],
            curr_rel,
        )

    def concept_filter_t(self, tail, relation):
        tail = self.ent2id[tail]
        relation = self.rel2id[relation]
        if str(relation) not in self.rel_t:
            return []
        rel_tc = self.rel_t[str(relation)]
        set_tc = set(rel_tc)
        t = []
        if self.rel2nn[str(relation)] == 2 or self.rel2nn[str(relation)] == 3:
            if tail in self.ent_conc:
                for conc in self.ent_conc[str(tail)]:
                    for ent in self.conc_ents[str(conc)]:
                        t.append(ent)
            else:
                for tc in rel_tc:
                    for ent in self.conc_ents[str(tc)]:
                        t.append(ent)
        else:
            if str(tail) in self.ent_conc:
                set_ent_conc = set(self.ent_conc[str(tail)])
            else:
                set_ent_conc = set([])
            set_diff = set_tc - set_ent_conc
            set_diff = list(set_diff)
            for conc in set_diff:
                for ent in self.conc_ents[str(conc)]:
                    t.append(ent)
        t = set(t)
        neg = []
        for item in list(t):
            neg.append(self.id2ent[item])

        return neg

