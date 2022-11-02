from common import *
from verifier.model import EntailmentClassifier
from prover.proof import ProofStep, Proof, InvalidProofStep
from prover.search import ProofGraph
import numpy as np
import os
import json
import torch
import itertools
import pytorch_lightning as pl
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    LogitsProcessor,
)
from prover.evaluate import evaluate_entailmentbank, evaluate_ruletaker


# Some handcrafted heuristics for constraining the predicted proof steps.
# They often make the proof graph less cluttered but do not improve the final performance.
# So we do not use them by default.
class PermutationInvarianceLogitsProcessor(LogitsProcessor):
    def __init__(
        self, num_beams: int, context: List[OrderedDict[str, str]], tokenizer: Any
    ) -> None:
        self.num_beams = num_beams
        self.context = context
        self.tokenizer = tokenizer
        self.semicolon_token_id = tokenizer.convert_tokens_to_ids(";")

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        generated_texts = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )

        batch_size = input_ids.size(0) // self.num_beams
        unique_premises: List[Set[Any]] = [set() for _ in range(batch_size)]

        for i, prefix in enumerate(generated_texts):
            if "->" in prefix:  # conclusion
                if prefix.count("->") > 1:
                    scores[i, :] = float("-inf")
                    continue
                concl = prefix.split("->")[1].strip()
                if concl == "hypothesis":
                    # Only ";" after "-> hypothesis".
                    s = scores[i, self.semicolon_token_id].item()
                    scores[i, :] = float("-inf")
                    scores[i, self.semicolon_token_id] = s
                elif ";" in concl:
                    # Must end after ";"
                    s = scores[i, self.tokenizer.eos_token_id].item()
                    scores[i, :] = float("-inf")
                    scores[i, self.tokenizer.eos_token_id] = s
                elif (
                    concl != ""
                    and not concl.startswith("int")
                    and not "int".startswith(concl)
                ):
                    # The conclusion is either the hypothesis or an intermediate.
                    scores[i, :] = float("-inf")
                elif "-> int" in prefix:
                    # Only one conclusion for fixed premises.
                    j = scores[i, :].argmax()
                    s = scores[i, j].item()
                    scores[i, :] = float("-inf")
                    scores[i, j] = s

            else:  # premises
                n = i // self.num_beams
                premises = tuple(sorted([p.strip() for p in prefix.split("&")]))
                if premises in unique_premises[n] or len(set(premises)) < len(premises):
                    scores[i, :] = float("-inf")
                    continue
                unique_premises[n].add(premises)

                tokens = prefix.split()
                for t in tokens[:-1]:
                    if t != "&" and re.fullmatch(r"(int|sent)\d+", t) == None:
                        scores[i, :] = float("-inf")
                    elif (
                        re.fullmatch(r"sent\d+", t) != None and t not in self.context[n]
                    ):
                        scores[i, :] = float("-inf")
                if len(tokens) >= 1:
                    t = tokens[-1]
                    if (
                        t != "&"
                        and re.fullmatch(r"(int|sent)\d+", t) == None
                        and not "sent".startswith(t)
                        and not "int".startswith(t)
                    ):
                        scores[i, :] = float("-inf")

        return scores


class EntailmentWriter(pl.LightningModule):
    def __init__(
        self,
        dataset: str,
        stepwise: bool,
        max_num_steps: int,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        topk: int,
        max_input_len: int,
        proof_search: bool,
        verifier_weight: float,
        verifier_ckpt: Optional[str] = None,
        oracle_prover: Optional[bool] = False,
        oracle_verifier: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.stepwise = stepwise
        self.max_num_steps = max_num_steps
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.topk = topk
        self.verifier_weight = verifier_weight
        self.proof_search = proof_search
        self.oracle_prover = oracle_prover
        self.oracle_verifier = oracle_verifier
        if stepwise and verifier_weight > 0:
            assert verifier_weight <= 1.0
            assert verifier_ckpt is not None
            self.verifiers = [
                EntailmentClassifier.load_from_checkpoint(verifier_ckpt)
            ]  # Avoid making the verifier a submodule.

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_input_len)
        if (
            model_name.startswith("t5-")
            or model_name.startswith("google/t5-v1_1-")
            or model_name.startswith("google/byt5-")
        ):
            self.seq2seq = T5ForConditionalGeneration.from_pretrained(model_name)
        elif model_name.startswith("facebook/bart-"):
            self.seq2seq = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            raise NotImplementedError

    def forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        return self.seq2seq(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss

    def move_verifier_to_device(self) -> None:
        if hasattr(self, "verifiers"):
            self.verifiers[0].to(self.device)

    def on_train_start(self) -> None:
        self.move_verifier_to_device()
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # type: ignore
            assert self.trainer is not None
            print(f"Logging to {self.trainer.log_dir}")

    def on_validation_start(self) -> None:
        self.move_verifier_to_device()

    def on_test_start(self) -> None:
        self.move_verifier_to_device()

    def generate_entire_proof(
        self, input_text: List[str]
    ) -> Tuple[List[str], List[float]]:
        """
        Single-shot proof generation with text-to-text transformers.
        """
        assert self.trainer is not None
        input = self.tokenizer(
            input_text,
            padding="longest",
            max_length=self.trainer.datamodule.max_input_len,  # type: ignore
            truncation=True,
            return_tensors="pt",
        )
        output = self.seq2seq.generate(
            input_ids=input.input_ids.to(self.device, non_blocking=True),
            attention_mask=input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.trainer.datamodule.max_output_len,  # type: ignore
            num_beams=self.num_beams,
            num_return_sequences=1,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        scores = output.sequences_scores.detach().exp().tolist()
        return output_text, scores

    def generate_stepwise_proof(
        self, proof_gt: List[Proof], batch_idx: int
    ) -> Tuple[List[str], List[float]]:
        """
        Stepwise proof generation.
        """
        proof_pred, step_scores = self.generate_greedy_proofs(proof_gt)
        if not self.proof_search:
            proof_text_pred = [pt.proof_text for pt in proof_pred]
            score = [min(s) if len(s) > 0 else 0.0 for s in step_scores]
        else:
            batch_size = len(proof_gt)
            proof_text_pred = []
            score = []
            for i in range(batch_size):
                p, s = self.search_proof(proof_gt[i], proof_pred[i], step_scores[i])
                proof_text_pred.append(p)
                score.append(s)
        return proof_text_pred, score

    def generate_proof_step(
        self,
        input_text: List[str],
    ) -> Tuple[List[str], Any]:
        """
        Generate a single proof step with text-to-text transformers.
        """
        input = self.tokenizer(
            input_text,
            padding="longest",
            max_length=self.trainer.datamodule.max_input_len,  # type: ignore
            truncation=True,
            return_tensors="pt",
        )

        output = self.seq2seq.generate(
            input_ids=input.input_ids.to(self.device, non_blocking=True),
            attention_mask=input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.trainer.datamodule.max_output_len,  # type: ignore
            num_beams=self.num_beams,
            num_return_sequences=self.topk,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        batch_size = len(input_text)
        assert len(output_text) % batch_size == 0
        k = len(output_text) // batch_size  # k predicted steps for each example.
        output_text = [output_text[i * k : (i + 1) * k] for i in range(batch_size)]

        output_scores = output.sequences_scores.detach().exp().cpu().numpy()
        assert 0.0 <= output_scores.min() <= output_scores.max() <= 1.0
        output_scores = [output_scores[i * k : (i + 1) * k] for i in range(batch_size)]

        return output_text, output_scores

    def generate_greedy_proofs(
        self, proof_gt: List[Proof]
    ) -> Tuple[List[Proof], List[List[float]]]:
        """
        Greedily stepwise proof generation.
        """
        all_proof_pred = [
            Proof(pt.context, pt.hypothesis, proof_text="", strict=True)
            for pt in proof_gt
        ]
        proof_pred = all_proof_pred
        unfinished_indexes = list(range(len(proof_gt)))
        all_step_scores: List[List[float]] = [[] for _ in proof_gt]

        for _ in range(self.max_num_steps):
            if len(unfinished_indexes) == 0:
                # All examples in the batch has been finished.
                break
            input_text = [
                f"$hypothesis$ = {pt.hypothesis} ; $context$ = {pt.serialize_context()} ; $proof$ = {'' if pt.proof_text == '' else pt.proof_text + ';'}"
                for pt in proof_pred
            ]
            output_text, output_scores = self.generate_proof_step(input_text)

            proof_steps, prover_scores = self.filter_invalid_steps(
                output_text,
                output_scores,
                proof_pred,
                strict=True,
            )

            scores = self.calculate_score(proof_steps, prover_scores, proof_gt)
            proof_steps = [
                steps[0] if len(steps) > 0 else None for steps in proof_steps
            ]
            scores = [s[0] if len(s) > 0 else 0.0 for s in scores]

            # Execute the predicted proof steps.
            finished_indexes = []
            for i, j in enumerate(unfinished_indexes):
                step = proof_steps[i]
                if step is None:
                    # Try to get some partial credits.
                    step = self.normalize_predicted_step(
                        output_text[i][0], proof_pred[i]
                    )
                    idx = step.find(";")
                    if idx != -1:
                        step = step[:idx]
                    try:
                        step = ProofStep(proof_pred[i], step, strict=False)
                        if step.is_final():
                            finished_indexes.append(i)
                        proof_pred[i].execute(step)
                        all_step_scores[j].append(float(output_scores[i][0]))
                    except InvalidProofStep:
                        finished_indexes.append(i)
                        proof_pred[i].proof_text = "INVALID_PROOF"
                else:
                    if step.is_final():
                        finished_indexes.append(i)
                    proof_pred[i].execute(step)
                    all_step_scores[j].append(scores[i])

            unfinished_indexes = [
                j for i, j in enumerate(unfinished_indexes) if i not in finished_indexes
            ]
            proof_pred = [
                pt for i, pt in enumerate(proof_pred) if i not in finished_indexes
            ]

        assert (
            pt.is_complete() or pt.proof_text == "INVALID_PROOF"
            for pt in all_proof_pred
        )
        return all_proof_pred, all_step_scores

    def generate_oracle_proof_step(
        self,
        input_text: List[str],
        proof_gt: Proof,
    ) -> Tuple[List[str], List[float]]:
        """
        Oracle prover.
        """
        output_text, output_scores = self.generate_proof_step(input_text)

        for i, inp in enumerate(input_text):
            _, partial_proof = inp.split("$proof$ = ")
            partial_proof = Proof(
                proof_gt.context,
                proof_gt.hypothesis,
                partial_proof,
                strict=True,
            )

            # Add all steps in proof_gt that are valid w.r.t. `partial_proof`.
            for step in proof_gt.proof_steps:
                for sent in step.premise_sents:
                    if sent not in partial_proof:
                        break
                else:
                    premise_idents = []
                    for ident, sent in zip(step.premise_idents, step.premise_sents):
                        if re.fullmatch(r"int\d+", ident):
                            ident = re.search(f"(?P<ident>int\d+): {sent}", inp)[
                                "ident"
                            ]
                        premise_idents.append(ident)
                    premises = " & ".join(premise_idents)
                    if step.conclusion_ident == "hypothesis":
                        conclusion = "hypothesis"
                    else:
                        conclusion = f"int: {step.conclusion_sent}"
                    text = f"{premises} -> {conclusion};"
                    output_text[i].append(text)
                    output_scores[i] = np.append(output_scores[i], 1.0)

        return output_text, output_scores

    def calculate_score(
        self,
        proof_steps: List[List[ProofStep]],
        prover_scores: List[List[float]],
        proof_gt: List[Proof],
    ) -> List[List[float]]:
        """
        Calculate the a step's score as the average between the prover score and the verifier score.
        """
        if self.verifier_weight == 0:
            return prover_scores

        batch_premises = []
        batch_conclusion = []
        batch_proof_gt = []

        for i, steps in enumerate(proof_steps):
            for s in steps:
                batch_premises.append(s.premise_sents)
                batch_conclusion.append(s.conclusion_sent)
                batch_proof_gt.append(proof_gt[i])

        if self.oracle_verifier:
            verifier_scores = self.calculate_oracle_verifier_score(
                batch_premises,
                batch_conclusion,
                batch_proof_gt,
            )
        else:
            verifier_scores = self.verifiers[0].batch_score(
                batch_premises, batch_conclusion
            )

        scores = []
        idx = 0
        for ps in prover_scores:
            if len(ps) == 0:
                scores.append([])
            else:
                scores.append(
                    (
                        (1.0 - self.verifier_weight) * np.array(ps)
                        + self.verifier_weight * verifier_scores[idx : idx + len(ps)]
                    ).tolist()
                )
                idx += len(ps)

        return scores

    def calculate_oracle_verifier_score(
        self,
        batch_premises: List[List[str]],
        batch_conclusion: List[str],
        batch_proof_gt: List[Proof],
    ) -> List[float]:
        """
        Oracle verifier.
        """
        verifier_scores = self.verifiers[0].batch_score(
            batch_premises, batch_conclusion
        )
        assert len(batch_premises) == len(batch_conclusion) == len(batch_proof_gt)

        for i, (premises, conclusion, proof_gt) in enumerate(
            zip(
                batch_premises,
                batch_conclusion,
                batch_proof_gt,
            )
        ):
            for step in proof_gt.proof_steps:
                if (
                    sorted(step.premise_sents) == sorted(premises)
                    and step.conclusion_sent == conclusion
                ):
                    verifier_scores[i] = 1.0
                    break

        return verifier_scores

    def search_proof(
        self,
        proof_gt: Proof,
        proof_greedy: Proof,
        step_scores_greedy: List[float],
    ) -> Tuple[str, float]:
        context, hypothesis = proof_gt.context, proof_gt.hypothesis
        pg = ProofGraph(context, hypothesis)
        pg.initialize(proof_greedy.proof_steps, step_scores_greedy)

        explored_proofs: Set[str] = set()
        context_text = proof_gt.serialize_context()

        while True:
            partial_proof = pg.sample_proof_tree(explored_proofs)
            if partial_proof is None:
                break
            explored_proofs.add(partial_proof)

            input_text = [
                f"$hypothesis$ = {hypothesis} ; $context$ = {context_text} ; $proof$ = {partial_proof}"
            ]
            if self.oracle_prover:
                output_text, output_scores = self.generate_oracle_proof_step(
                    input_text, proof_gt
                )
            else:
                output_text, output_scores = self.generate_proof_step(input_text)
            proof_steps, prover_scores = self.filter_invalid_steps(
                output_text,
                output_scores,
                [Proof(context, hypothesis, partial_proof, strict=False)],
                strict=False,
            )
            scores = self.calculate_score(
                proof_steps,
                prover_scores,
                [proof_gt],
            )

            proof_steps = list(itertools.chain.from_iterable(proof_steps))
            scores = list(itertools.chain.from_iterable(scores))

            graph_updated = False
            for ps, s in zip(proof_steps, scores):
                if pg.expand(ps, s):
                    graph_updated = True
            if not graph_updated:
                break

        proof = pg.extract_proof("hypothesis", rename=True)
        return proof, pg.graph.nodes["hypothesis"]["score"]

    def normalize_predicted_step(self, step: str, proof: Proof) -> str:
        if "-> int:" in step:
            step = step.replace("-> int:", f"-> {proof.next_int()}:").strip()
        return step

    def filter_invalid_steps(
        self,
        output_text: List[str],
        output_scores: List[float],
        proofs: List[Proof],
        strict: bool,
    ) -> Tuple[List[List[ProofStep]], List[List[float]]]:
        batch_size = len(proofs)

        all_proof_steps = [[] for _ in range(batch_size)]
        all_scores = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            assert len(output_text[i]) == len(output_scores[i])

            for text, score in zip(output_text[i], output_scores[i]):
                idx = text.find(";")
                if idx != -1:
                    text = text[:idx]
                else:
                    continue
                s = self.normalize_predicted_step(text, proofs[i])
                try:
                    step = ProofStep(proofs[i], s, strict)
                except InvalidProofStep:
                    continue
                all_proof_steps[i].append(step)
                all_scores[i].append(float(score))

        return all_proof_steps, all_scores

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        if self.stepwise:
            loss = self(
                batch["input_seq_ids"], batch["input_seq_mask"], batch["output_seq_ids"]
            )
            self.log("loss_train", loss, on_epoch=True, sync_dist=True)
        else:
            loss = self(
                batch["input_seq_ids"],
                batch["input_seq_mask"],
                batch["output_seq_ids"],
            )
            self.log("loss_train", loss, on_epoch=True, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch: Batch, batch_idx: int) -> Tuple[Any]:  # type: ignore
        return self.val_test_step("val", batch, batch_idx)

    def test_step(self, batch: Batch, batch_idx: int) -> Tuple[Any]:  # type: ignore
        return self.val_test_step("test", batch, batch_idx)

    def validation_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("val", outputs)

    def test_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("test", outputs)

    def val_test_step(self, split: str, batch: Batch, batch_idx: int) -> Tuple[Any]:
        if self.stepwise:
            proof_pred, score = self.generate_stepwise_proof(batch["proof"], batch_idx)
        else:
            loss = self(
                batch["input_seq_ids"],
                batch["input_seq_mask"],
                batch["output_seq_ids"],
            )
            self.log(f"loss_{split}", loss, sync_dist=True)
            proof_pred, score = self.generate_entire_proof(batch["input_seq"])

        if self.dataset == "entailmentbank":
            return proof_pred, score, batch["proof"]
        else:
            return (
                proof_pred,
                score,
                batch["proof"],
                batch["answer"],
                batch["depth"],
                batch["all_proofs"],
            )

    def val_test_epoch_end(self, split: str, outputs: Iterable[Any]) -> None:
        results = []

        for out in outputs:
            if self.dataset == "entailmentbank":
                for proof_pred, score, proof in zip(*out):
                    results.append(
                        {
                            "proof_pred": proof_pred,
                            "score": score,
                            "hypothesis": proof.hypothesis,
                            "context": proof.context,
                            "proof_gt": proof.proof_text,
                        }
                    )
            else:
                for proof_pred, score, proof, answer, depth, all_proofs in zip(*out):
                    results.append(
                        {
                            "answer": answer,
                            "depth": depth,
                            "all_proofs": all_proofs,
                            "proof_pred": proof_pred,
                            "score": score,
                            "hypothesis": proof.hypothesis,
                            "context": proof.context,
                            "proof_gt": proof.proof_text,
                        }
                    )

        assert self.trainer is not None
        if self.logger is not None and self.trainer.log_dir is not None:
            json_path = os.path.join(self.trainer.log_dir, f"results_{split}.json")
            json.dump(results, open(json_path, "wt"))
            if self.dataset == "entailmentbank":
                tsv_path = os.path.join(self.trainer.log_dir, f"results_{split}.tsv")
                with open(tsv_path, "wt") as oup:
                    for r in results:
                        proof = r["proof_pred"].strip()
                        if not proof.endswith(";"):
                            proof += ";"
                        oup.write(f"$proof$ = {proof}\n")
                print(f"Validation results saved to {json_path} and {tsv_path}")
            else:
                print(f"Validation results saved to {json_path}")

        if self.dataset == "entailmentbank" and results[0]["proof_gt"] != "":
            em, f1 = evaluate_entailmentbank(results, eval_intermediates=False)
            for k, v in em.items():
                self.log(f"ExactMatch_{k}_{split}", v, on_step=False, on_epoch=True)
            for k, v in f1.items():
                self.log(f"F1_{k}_{split}", v, on_step=False, on_epoch=True)

        elif self.dataset == "ruletaker":
            answer_accuracies, proof_accuracies = evaluate_ruletaker(results)
            for k in answer_accuracies.keys():
                self.log(
                    f"Accuracy_answer_{k}_{split}",
                    answer_accuracies[k],
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"Accuracy_proof_{k}_{split}",
                    proof_accuracies[k],
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self) -> Dict[str, Any]:
        assert self.trainer is not None
        if self.trainer.max_steps != -1:
            max_steps = self.trainer.max_steps
        else: 
            max_steps = (
                self.trainer.max_epochs
                * len(self.trainer.datamodule.train_dataloader())  # type: ignore
                // self.trainer.accumulate_grad_batches
            )
        return get_optimizers(
            self.parameters(),
            self.lr,
            self.warmup_steps,
            max_steps,
        )
