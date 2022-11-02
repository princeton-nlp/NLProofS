from common import *
from copy import deepcopy
import itertools
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import json
import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi


def sample_similar_sentence(query: str, corpus: List[str]) -> str:
    # Sample a sentence in `corpus` that is similar to `query`.
    assert query not in corpus
    tokenized_query = query.split()
    tokenized_corpus = [sent.split() for sent in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = F.softmax(torch.tensor(bm25.get_scores(tokenized_query)), dim=0)
    dist = Categorical(probs=scores)
    sent = corpus[dist.sample()]
    return sent  # type: ignore


def enumerate_premise_nodes(node: TreeNode, max_num: int) -> List[List[TreeNode]]:
    all_premises: List[List[TreeNode]] = [[]]

    for child in node.children:
        if child.is_leaf():
            for premises in all_premises:
                premises.append(child)
        else:
            prev_all_premises = all_premises
            all_premises = []
            for premises in prev_all_premises:
                all_premises.append(premises + [child])
            for child_premises in enumerate_premise_nodes(child, max_num - 1):
                for premises in prev_all_premises:
                    all_premises.append(premises + child_premises)

    return [premises for premises in all_premises if len(premises) <= max_num]


def powerset(iterable: Iterable[Any]) -> List[Tuple[Any]]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(
        itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)  # type: ignore
        )
    )


class EntailmentDataset(Dataset):  # type: ignore
    def __init__(
        self,
        path: str,
        model_name: str,
        max_num_premises: int,
        split: str,
        max_input_len: int,
        irrelevant_distractors_only: bool,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_input_len)
        assert split in ("train", "val")
        self.split = split
        self.max_num_premises = max_num_premises  # The maximum number of premises used in data augmentation.
        self.max_input_len = max_input_len
        self.irrelevant_distractors_only = irrelevant_distractors_only
        self.data = self.preprocess(path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]
        premises = deepcopy(ex["premises"])
        random.shuffle(premises)
        premises = ". ".join(premises) + "."
        return {
            "premises": premises,
            "conclusion": ex["conclusion"],
            "label": ex["label"],  # Whether it is a valid entailment.
        }

    def preprocess(self, path: str) -> List[Example]:
        raise NotImplementedError

    def collate(self, examples: List[Example]) -> Batch:
        entailment = self.tokenizer(
            [(ex["premises"], ex["conclusion"]) for ex in examples],
            padding="longest",
            truncation="longest_first",
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        label = torch.tensor([ex["label"] for ex in examples], dtype=torch.int64)
        return {
            "premises": [ex["premises"] for ex in examples],
            "conclusion": [ex["conclusion"] for ex in examples],
            "input_ids": entailment["input_ids"],
            "attention_mask": entailment["attention_mask"],
            "label": label,
        }


class EntailmentBankDataset(EntailmentDataset):
    def preprocess(self, path: str) -> List[Example]:
        """
        Extract positive and negative examples from ground truth proof trees.
        """
        data = []
        num_pos = 0
        num_neg = 0

        for line in open(path):
            ex = json.loads(line)
            context = extract_context(ex["context"])
            pos, neg = self.extract_examples(ex, context)
            data.extend(pos)
            data.extend(neg)
            num_pos += len(pos)
            num_neg += len(neg)

        random.shuffle(data)
        print(f"#positives: {num_pos}\n#pseudo-negatives: {num_neg}")

        return data

    def extract_examples(
        self, ex: Example, context: OrderedDict[str, str]
    ) -> Tuple[List[Example], List[Example]]:
        """
        Extract positive and negative examples from a proof tree.
        """
        positives = []
        negatives = []
        tree = deserialize(ex["hypothesis"], context, ex["proof"])

        def create_positive(premises: List[str], conclusion: str) -> None:
            assert len(premises) >= 2
            positives.append(
                {"premises": premises, "conclusion": conclusion, "label": True}
            )

        def create_negative(premises: List[str], conclusion: str) -> None:
            assert len(premises) >= 2
            negatives.append(
                {"premises": premises, "conclusion": conclusion, "label": False}
            )

        for node in tree.traverse():
            if node.is_leaf():
                continue

            if self.split != "train":
                premises = [child.sent for child in node.children]
                if len(premises) >= 2:
                    create_positive(premises, node.sent)

            else:
                # 1. Enumerate all combinations of premises leading to node.sent.
                for premise_nodes in enumerate_premise_nodes(
                    node, self.max_num_premises
                ):
                    premises = [pn.sent for pn in premise_nodes]
                    num_premises = len(premises)

                    if num_premises >= 2:
                        create_positive(premises, node.sent)

                        # 2. Perturbe them to generate negatives.
                        for i, p in enumerate(premises):
                            if self.irrelevant_distractors_only:
                                candidates = [
                                    sent
                                    for sent in context.values()
                                    if sent not in premises
                                ]
                            else:
                                candidates = [
                                    sent for sent in context.values() if sent != p
                                ]
                            alternative = sample_similar_sentence(p, candidates)
                            prems = deepcopy(premises)
                            prems[i] = alternative
                            create_negative(prems, node.sent)

                        if num_premises > 2:
                            for subset in powerset(premises):
                                if 2 <= len(subset) < num_premises:
                                    create_negative(list(subset), node.sent)

        if self.split == "train":
            # Copy premises.
            leaf_sents = [node.sent for node in tree.get_leaves()]
            for s1 in leaf_sents:
                for s2 in leaf_sents:
                    if s1 == s2:
                        continue
                    create_negative([s1, s2], s1)

        return positives, negatives


class RuleTakerDataset(EntailmentDataset):
    def preprocess(self, path: str) -> List[Example]:
        """
        Extract positive and negative examples from ground truth proof trees.
        """
        data = []

        for line in open(path):
            ex = json.loads(line)
            pos, neg = self.extract_examples(ex)
            data.extend(pos)
            data.extend(neg)

        data = list(set(data))
        random.shuffle(data)
        data = [
            {"premises": list(ex[0]), "conclusion": ex[1], "label": ex[2]}
            for ex in data
        ]
        num_pos = sum([1 for x in data if x["label"] == True])
        num_neg = sum([1 for x in data if x["label"] == False])
        print(f"#positives: {num_pos}\n#pseudo-negatives: {num_neg}")

        return data

    def extract_examples(
        self,
        ex: Example,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Extract positive and negative examples from a proof tree.
        """
        context = extract_context(ex["context"])
        positives = []
        negatives = []

        def create_positive(premises: List[str], conclusion: str) -> None:
            positives.append((tuple(premises), conclusion, True))

        def create_negative(premises: List[str], conclusion: str) -> None:
            negatives.append((tuple(premises), conclusion, False))

        for proof in ex["proofs"]:
            tree = deserialize(ex["hypothesis"], context, proof)
            if tree is None:
                assert proof == ""
                continue

            for node in tree.traverse():
                if node.is_leaf():
                    continue

                premises = [child.sent for child in node.children]
                if ex["answer"] == False and node.is_root():
                    create_negative(premises, node.sent)
                    continue
                else:
                    create_positive(premises, node.sent)

                if self.split == "train":
                    if "does not " in node.sent:
                        create_negative(premises, node.sent.replace("does not ", ""))
                    elif "do not " in node.sent:
                        create_negative(premises, node.sent.replace("do not ", ""))
                    elif "cannot " in node.sent:
                        create_negative(premises, node.sent.replace("cannot ", "can"))
                    elif "not " in node.sent:
                        create_negative(premises, node.sent.replace("not ", ""))

                    if node.sent.startswith("i don't think "):
                        create_negative(
                            premises, node.sent.replace("i don't think ", "")
                        )
                    else:
                        create_negative(premises, f"i don't think {node.sent}")

                    for i, p in enumerate(premises):
                        if self.irrelevant_distractors_only:
                            candidates = [
                                sent
                                for sent in context.values()
                                if sent not in premises
                            ]
                        else:
                            candidates = [
                                sent for sent in context.values() if sent != p
                            ]
                        if len(candidates) == 0:
                            continue
                        alternative = sample_similar_sentence(p, candidates)
                        prems = deepcopy(premises)
                        prems[i] = alternative
                        create_negative(prems, node.sent)

        return positives, negatives


class EntailmentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        path_train: str,
        path_val: str,
        model_name: str,
        batch_size: int,
        num_workers: int,
        max_num_premises: int,
        max_input_len: int,
        irrelevant_distractors_only: bool,
    ) -> None:
        super().__init__()
        if dataset == "entailmentbank":
            self.Dataset = EntailmentBankDataset
        elif dataset == "ruletaker":
            self.Dataset = RuleTakerDataset  # type: ignore
        else:
            raise NotImplementedError
        self.path_train = path_train
        self.path_val = path_val
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_premises = max_num_premises
        self.max_input_len = max_input_len
        self.irrelevant_distractors_only = irrelevant_distractors_only

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = self.Dataset(
                self.path_train,
                self.model_name,
                self.max_num_premises,
                "train",
                self.max_input_len,
                self.irrelevant_distractors_only,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = self.Dataset(
                self.path_val,
                self.model_name,
                self.max_num_premises,
                "val",
                self.max_input_len,
                self.irrelevant_distractors_only,
            )

    def train_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_train,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_val,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            pin_memory=True,
            drop_last=False,
        )
