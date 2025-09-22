from queue import PriorityQueue, Empty
import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)


class BestFirstWordSwapWIR(SearchMethod):
    def __init__(self, wir_method="unk", unk_token="[UNK]"):
        self.wir_method = wir_method
        self.unk_token = unk_token
        self.index_performance_history = {}

    def _update_index_scores(self, indices, scores, results):
        max_index = max(indices)
        if max_index >= len(scores):
            scores = np.concatenate([scores, np.zeros(max_index - len(scores) + 1)])

        for idx, score, result in zip(indices, scores, results):
            if idx not in self.index_performance_history:
                self.index_performance_history[idx] = []

            self.index_performance_history[idx].append(score - result.score if result else 0)

            if len(self.index_performance_history[idx]) > 5: 
                self.index_performance_history[idx] = self.index_performance_history[idx][-5:]
            adjusted_score = np.mean(self.index_performance_history[idx])
            scores[idx] += adjusted_score

    def _get_index_order(self, initial_text, max_len=-1):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""

        len_text, indices_to_order = self.get_indices_to_order(initial_text)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "weighted-saliency":
            # first, compute word saliency
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()

            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in indices_to_order:
                # Exit Loop when search_over is True - but we need to make sure delta_ps
                # is the same size as softmax_saliency_scores
                if search_over:
                    delta_ps = delta_ps + [0.0] * (
                        len(softmax_saliency_scores) - len(delta_ps)
                    )
                    break

                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, search_over = self.get_goal_results(
                    transformed_text_candidates
                )
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores * np.array(delta_ps)
            self._update_index_scores(indices_to_order, index_scores, leave_one_results)

        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])
            self._update_index_scores(indices_to_order, index_scores, leave_one_results)

        elif self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            index_scores = np.zeros(len_text)
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, index in enumerate(indices_to_order):
                matched_tokens = word2token_mapping[index]
                if not matched_tokens:
                    index_scores[i] = 0.0
                else:
                    agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    index_scores[i] = np.linalg.norm(agg_grad, ord=1)

            search_over = False

        elif self.wir_method == "random":
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]

        return index_order, search_over

    def perform_search(self, initial_result):
        priority_queue = PriorityQueue()
        priority_queue.put((-initial_result.score, initial_result))

        best_result = initial_result
        searched_texts = set()
        searched_texts.add(initial_result.attacked_text)

        while not priority_queue.empty():
            _, current_result = priority_queue.get()

            if current_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                return current_result

            index_order, search_over = self._get_index_order(current_result.attacked_text)
            i = 0
            while i < len(index_order) and not search_over:
                transformed_text_candidates = self.get_transformations(
                    current_result.attacked_text,
                    original_text=initial_result.attacked_text,
                    indices_to_modify=[index_order[i]],
                )
                i += 1

                if not transformed_text_candidates:
                    continue

                results, search_over = self.get_goal_results(transformed_text_candidates)
                for result in results:
                    if result.attacked_text not in searched_texts and result.score > best_result.score:
                        priority_queue.put((-result.score, result))
                        searched_texts.add(result.attacked_text)
                        best_result = result

                if search_over:
                    return best_result

        return best_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]


