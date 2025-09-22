import numpy as np
import torch
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import transformation_consists_of_word_swaps_and_deletions
from torch.nn.functional import softmax


class BeamAnnealingSearch(SearchMethod):
    def __init__(self, k0=2, k_min=2, k_max=6, T0=1.0, gamma=0.3, pe=0.9,
                 delta=1, scaling_factor=3, unk_token="[UNK]", wir_method="weighted-saliency"):
        """Initialize Beam Annealing Search algorithm with predefined hyperparameters"""
        self.k_min = k_min  # Minimum beam width
        self.k_max = k_max  # Maximum beam width
        self.T0 = T0  # Initial temperature
        self.gamma = gamma  # Cooling factor
        self.pe = pe  # Elitism probability
        self.delta = delta  # Smoothing factor
        self.unk_token = unk_token  # Token used for perturbation
        self.wir_method = wir_method  # Word importance ranking method
        self.scaling_factor = scaling_factor
        self.k = k0  # Current beam width
        self.T = T0  # Current temperature

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
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        i = 0
        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED and i < len(index_order):
            potential_next_beam = []
            # 1. Generate mutants for each text in the beam
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text,
                    indices_to_modify=[index_order[i]]
                )
                original_score = self.get_goal_results([text])[0][0].score
                for transformed_text in transformations:
                    transformed_score = self.get_goal_results([transformed_text])[0][0].score
                    # Simulated annealing strategy: decide whether to accept the transformation
                    if transformed_score > original_score:
                        potential_next_beam.append(transformed_text)  # Better solution, accept it directly
                    else:
                        accept_prob = np.exp(-(transformed_score - original_score) / self.T)
                        if np.random.rand() < accept_prob:
                            potential_next_beam.append(transformed_text)  # Accept with probability P(accept)
            i += 1
            if len(potential_next_beam) == 0:
                return best_result
            # Calculate the score of each candidate and select the best individual
            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            # Get the current best score
            current_best_score = best_result.score
            # Get the best score among the current potential candidates
            potential_best_score = np.max(scores)
            # Replace only if the potential best score is higher than the current best score
            if potential_best_score > current_best_score:
                best_result = results[scores.argmax()]
            if search_over:
                return best_result
            s_star = results[scores.argmax()].attacked_text
            # Calculate information gain and dynamically adjust beam width
            H_B = self.information_gain(scores)
            self.k = int(max(self.k_min, min(self.k_max, self.k * (1 + H_B / self.k_max), self.k + self.delta)))
            P_select_s_star = np.exp(-potential_best_score) / np.sum(np.exp(-scores))
            # Calculate the probability of retaining s_star
            retain_prob = self.pe + (1 - self.pe) * P_select_s_star
            # Use this probability to decide whether to keep s_star
            new_beam = [s_star] if np.random.rand() < retain_prob else []
            remaining_candidates = [s for s in potential_next_beam if s != s_star]
            # 6. Update the beam and continue the next search round
            # Calculate the selection probability P_select(s)
            # Get the score of remaining_candidates
            remaining_scores = np.array([r.score for r in results if r.attacked_text in remaining_candidates])
            # Add best_result to remaining_candidates
            if best_result.attacked_text not in remaining_candidates:
                remaining_candidates.append(best_result.attacked_text)
                remaining_scores = np.append(remaining_scores, best_result.score)
            remaining_scores = np.maximum(remaining_scores, 1e-5)
            unnormalized_probs = np.power(remaining_scores, self.scaling_factor)
            # Normalization makes P_select(s) a probability distribution
            P_select = unnormalized_probs / np.sum(unnormalized_probs)
            num_remaining = min(self.k - len(new_beam), len(remaining_candidates))
            if num_remaining > 0:
                sampled_candidates = np.random.choice(
                    remaining_candidates, num_remaining, replace=False, p=P_select
                )
                new_beam += list(sampled_candidates)
            beam = new_beam
            # 7. Update temperature
            self.T = self.update_temperature(i)

        return best_result

    def information_gain(self, scores):
        """Compute Shannon entropy for beam width adjustment"""
        P = scores / np.sum(scores)
        H_B = -np.sum(P * np.log(P + 1e-10))
        return H_B

    def update_temperature(self, t):
        """Temperature cooling with logarithmic decay"""
        return self.T0 / (1 + self.gamma * np.log(1 + t))

    def check_transformation_compatibility(self, transformation):
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["wir_method"]
