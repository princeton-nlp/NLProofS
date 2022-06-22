"""
Dataloading for EntailmentBank and RuleTaker.
"""
from copy import deepcopy
from common import *
from prover.proof import Proof, InvalidProofStep
import random
import json
import itertools
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer


def read_entailmentbank_proofs(path: str, is_train: bool) -> List[Example]:
    """
    Load the EntailmentBank dataset.
    """
    data = []
    num_invalid = 0

    for line in open(path):
        ex = json.loads(line)
        hypothesis = normalize(ex["hypothesis"])
        context = extract_context(ex["context"])
        proof_text = normalize(ex["proof"].strip())
        try:
            proof = Proof(
                context,
                hypothesis,
                proof_text,
                strict=is_train,
                requires_complete=is_train,
            )
            data.append({"proof": proof})
        except InvalidProofStep:
            assert is_train
            num_invalid += 1

    print(f"{len(data)} proofs loaded. {num_invalid} invalid ones removed.")
    return data


def read_ruletaker_proofs(path: str, is_train: bool) -> List[Example]:
    """
    Load the RuleTaker dataset.
    """
    data = []
    num_invalid = 0

    for line in open(path):
        ex = json.loads(line)
        hypothesis = normalize(ex["hypothesis"])
        context = extract_context(ex["context"])

        if is_train:
            for proof in ex["proofs"]:
                try:
                    prf = Proof(
                        context,
                        hypothesis,
                        normalize(proof.strip()),
                        strict=True,
                        requires_complete=True,
                    )
                    ans = ex["answer"]
                    assert ans == True
                    data.append(
                        {
                            "answer": ans,
                            "depth": ex["depth"],
                            "proof": prf,
                            "all_proofs": ex["proofs"],
                        }
                    )
                except InvalidProofStep:
                    num_invalid += 1

        else:
            proof = ex["proofs"][0] if len(ex["proofs"]) > 0 else ""
            data.append(
                {
                    "answer": ex["answer"],
                    "depth": ex["depth"],
                    "proof": Proof(
                        context,
                        hypothesis,
                        proof,
                        strict=False,
                        requires_complete=False,
                    ),
                    "all_proofs": ex["proofs"],
                }
            )
            data.append(
                {
                    "answer": not ex["answer"] if ex["answer"] != "Unknown" else "Unknown",
                    "depth": ex["depth"],
                    "proof": Proof(
                        context,
                        f"i don't think {hypothesis}",
                        proof,
                        strict=False,
                        requires_complete=False,
                    ),
                    "all_proofs": ex["proofs"],
                }
            )

    print(f"{len(data)} proofs loaded. {num_invalid} invalid ones removed.")
    return data


def collect_proved_subtrees(tree: TreeNode, prob: float) -> Iterable[TreeNode]:
    if tree.is_leaf():
        return []
    elif random.random() < prob:
        return [tree]
    else:
        return itertools.chain.from_iterable(
            collect_proved_subtrees(child, prob) for child in tree.children
        )


class EntireProofsDataset(Dataset):  # type: ignore
    def __init__(
        self,
        dataset: str,
        path: str,
        model_name: str,
        max_input_len: int,
        max_output_len: int,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.is_train = is_train
        if dataset == "entailmentbank":
            self.data = read_entailmentbank_proofs(path, is_train)
        else:
            assert dataset == "ruletaker"
            self.data = read_ruletaker_proofs(path, is_train)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]
        proof = ex["proof"]
        if self.is_train:
            proof = proof.shuffle_context()

        input_seq = f"$hypothesis$ = {proof.hypothesis} ; $context$ = {proof.serialize_context()}"

        ex = deepcopy(ex)
        ex["input_seq"] = input_seq
        ex["output_seq"] = proof.proof_text
        return ex

    def collate(self, examples: List[Example]) -> Batch:
        inp = [ex["input_seq"] for ex in examples]
        input_seq = self.tokenizer(
            inp,
            padding="longest",
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt",
        )

        oup = [ex["output_seq"] for ex in examples]
        output_seq = self.tokenizer(
            oup,
            padding="longest",
            max_length=self.max_output_len,
            truncation=True,
            return_tensors="pt",
        )
        output_seq.input_ids[output_seq.input_ids == self.tokenizer.pad_token_id] = -100

        batch = {
            "input_seq": inp,
            "input_seq_ids": input_seq.input_ids,
            "input_seq_mask": input_seq.attention_mask,
            "output_seq": oup,
            "output_seq_ids": output_seq.input_ids,
            "output_seq_mask": output_seq.attention_mask,
        }
        for k in examples[0].keys():
            if k not in ("input_seq", "output_seq"):
                batch[k] = [ex[k] for ex in examples]
        return batch


class StepwiseDataset(Dataset):  # type: ignore
    def __init__(
        self,
        dataset: str,
        path: str,
        model_name: str,
        max_input_len: int,
        max_output_len: int,
        sample_goal: str,
        subtree_proved_prob: float,
        subtree_proved_all_or_none: bool,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.sample_goal = sample_goal
        self.subtree_proved_prob = subtree_proved_prob
        self.subtree_proved_all_or_none = subtree_proved_all_or_none
        self.is_train = is_train
        if dataset == "entailmentbank":
            self.data = read_entailmentbank_proofs(path, is_train)
        else:
            assert dataset == "ruletaker"
            self.data = read_ruletaker_proofs(path, is_train)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]
        if self.is_train:
            return self.get_example_train(ex)
        else:
            return self.get_example_eval(ex)

    def collate(self, examples: List[Example]) -> Batch:
        inp = [ex["input_seq"] for ex in examples]
        input_seq = self.tokenizer(
            inp,
            padding="longest",
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt",
        )

        batch = {
            "input_seq": inp,
            "input_seq_ids": input_seq.input_ids,
            "input_seq_mask": input_seq.attention_mask,
        }
        for k in examples[0].keys():
            if k not in ("input_seq", "output_seq"):
                batch[k] = [ex[k] for ex in examples]

        if self.is_train:
            oup = [ex["output_seq"] for ex in examples]
            output_seq = self.tokenizer(
                oup,
                padding="longest",
                max_length=self.max_output_len,
                truncation=True,
                return_tensors="pt",
            )
            output_seq.input_ids[
                output_seq.input_ids == self.tokenizer.pad_token_id
            ] = -100
            batch["output_seq"] = oup
            batch["output_seq_ids"] = output_seq.input_ids
            batch["output_seq_mask"] = output_seq.attention_mask

        return batch

    def get_example_train(self, ex: Example) -> Example:
        proof = ex["proof"]
        if self.is_train:
            proof = proof.shuffle_context()

        # Sample the proof step.
        tree = proof.to_tree()
        int_node = random.choice(get_internal_nodes(tree))

        # Sample the goal.
        if self.sample_goal == "hypothesis":
            goal_node = tree.get_tree_root()
        else:
            assert self.sample_goal == "intermediates"
            ancestors = int_node.get_ancestors()
            assert int_node not in ancestors
            ancestors.append(int_node)
            goal_node = random.choice(ancestors)

        # Sample the partial proof.
        proved_subtrees = [node for node in int_node.children if not node.is_leaf()]
        if int_node is not goal_node:
            unproved_child = int_node
            for node in int_node.iter_ancestors():
                for child in node.children:
                    if child is unproved_child or child.is_leaf():
                        continue
                    if self.subtree_proved_all_or_none:
                        if random.random() < self.subtree_proved_prob:
                            proved_subtrees.append(child)
                    else:
                        proved_subtrees.extend(
                            collect_proved_subtrees(child, self.subtree_proved_prob)
                        )
                if node is goal_node:
                    break
                else:
                    unproved_child = node
        proved_subtrees.reverse()
        random.shuffle(proved_subtrees)
        partial_proof = " ".join(serialize(t) for t in proved_subtrees)

        # goal_context
        input_seq = f"$hypothesis$ = {goal_node.sent} ; $context$ = {proof.serialize_context()} ; $proof$ = {partial_proof}"

        premises = [node.name for node in int_node.children]
        random.shuffle(premises)
        output_seq = " & ".join(premises)
        if goal_node is int_node:
            output_seq = output_seq + " -> hypothesis;"
        else:
            output_seq = output_seq + f" -> int: {int_node.sent};"

        ex = deepcopy(ex)
        ex["proof"] = proof
        ex["input_seq"] = input_seq
        ex["output_seq"] = output_seq
        return ex

    def get_example_eval(self, ex: Example) -> Example:
        proof = ex["proof"]
        context_text = proof.serialize_context()
        input_seq = f"$hypothesis$ = {proof.hypothesis} ; $context$ = {context_text} ; $proof$ = "

        ex = deepcopy(ex)
        ex["input_seq"] = input_seq
        return ex


class ProofDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        stepwise: bool,
        sample_goal: str,
        model_name: str,
        max_input_len: int,
        max_output_len: int,
        batch_size: int,
        num_workers: int,
        path_train: str,
        path_val: str,
        path_test: str,
        subtree_proved_prob: float,
        subtree_proved_all_or_none: bool,
    ) -> None:
        super().__init__()
        assert dataset in ("entailmentbank", "ruletaker")
        self.dataset = dataset
        self.stepwise = stepwise
        self.sample_goal = sample_goal
        self.model_name = model_name
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test
        self.subtree_proved_prob = subtree_proved_prob
        self.subtree_proved_all_or_none = subtree_proved_all_or_none

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self.stepwise:
                self.ds_train = StepwiseDataset(
                    self.dataset,
                    self.path_train,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
                    self.sample_goal,
                    self.subtree_proved_prob,
                    self.subtree_proved_all_or_none,
                    is_train=True,
                )
            else:
                self.ds_train = EntireProofsDataset(  # type: ignore
                    self.dataset,
                    self.path_train,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
                    is_train=True,
                )

        if stage in (None, "fit", "validate"):
            if self.stepwise:
                self.ds_val = StepwiseDataset(
                    self.dataset,
                    self.path_val,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
                    self.sample_goal,
                    self.subtree_proved_prob,
                    self.subtree_proved_all_or_none,
                    is_train=False,
                )
            else:
                self.ds_val = EntireProofsDataset(  # type: ignore
                    self.dataset,
                    self.path_val,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
                    is_train=False,
                )

        if stage in (None, "test"):
            if self.stepwise:
                self.ds_test = StepwiseDataset(
                    self.dataset,
                    self.path_test,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
                    self.sample_goal,
                    self.subtree_proved_prob,
                    self.subtree_proved_all_or_none,
                    is_train=False,
                )
            else:
                self.ds_test = EntireProofsDataset(  # type: ignore
                    self.dataset,
                    self.path_test,
                    self.model_name,
                    self.max_input_len,
                    self.max_output_len,
                    is_train=False,
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

    def test_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_test,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.ds_test.collate,
            pin_memory=True,
            drop_last=False,
        )
