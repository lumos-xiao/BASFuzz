import numpy as np
import torch
from torch.nn.functional import softmax
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)


class BeamWordSwapWIR(SearchMethod):

    def __init__(self, wir_method="unk", unk_token="[UNK]", beam_width=8, min_beam_width=1, P_max=1.0):
        self.wir_method = wir_method
        self.unk_token = unk_token
        self.beam_width = beam_width
        self.max_beam_width = beam_width
        self.min_beam_width = min_beam_width
        self.P_max = P_max

    def calculate_beam_width(self, P_s):
        P_s_normalized = P_s / self.P_max if self.P_max > 0 else 0
        return self.min_beam_width + (self.max_beam_width - self.min_beam_width) * P_s_normalized

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

        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

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
        beam = [initial_result.attacked_text]
        best_result = initial_result
        historical_best = initial_result
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        iter_num = 0
        prev_scores = {text: initial_result.score for text in beam}
        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED and iter_num < len(index_order):
            current_historical_best = historical_best
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text,
                    indices_to_modify=[index_order[iter_num]]
                )
                potential_next_beam += transformations

            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result
            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()]
            if search_over:
                return best_result
            if iter_num == 0:
                # Refill the beam. This works by sorting the scores
                # in descending order and filling the beam from there.
                best_indices = (-scores).argsort()[: self.beam_width]
                beam = [potential_next_beam[i] for i in best_indices]
            else:
                successful_hypotheses = sum(
                    scores[i] < prev_scores[beam[i]] if beam[i] in prev_scores else 0
                    for i in range(len(beam))
                )

                P_s = successful_hypotheses / len(potential_next_beam) if len(
                    potential_next_beam) > 0 else 0
                if P_s > self.P_max or self.P_max == 1.0:
                    self.P_max = P_s
                self.beam_width = int(self.calculate_beam_width(P_s))
                # Refill the beam. This works by sorting the scores
                # in descending order and filling the beam from there.
                best_indices = (-scores).argsort()[: self.beam_width]
                beam = [potential_next_beam[i] for i in best_indices]

            prev_scores = {beam[i]: scores[best_indices[i]] for i in range(len(beam))}
            if beam and current_historical_best.score > scores[best_indices[-1]]:
                beam[-1] = current_historical_best.attacked_text
                prev_scores[beam[-1]] = current_historical_best.score

            if best_result.score > historical_best.score:
                historical_best = best_result
            iter_num += 1

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
