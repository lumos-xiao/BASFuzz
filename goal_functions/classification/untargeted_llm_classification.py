"""

Determine successful in untargeted_LLM_Classification
----------------------------------------------------
"""

from .classification_goal_function import ClassificationGoalFunction
import logging


class UntargetedLLMClassification(ClassificationGoalFunction):

    def __init__(self, *args, inference, logger, target_max_score=None, **kwargs):
        self.target_max_score = target_max_score
        self.inference = inference
        self.test_name = logger
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        if self.target_max_score:
            return model_output[self.ground_truth_output] < self.target_max_score
        elif (model_output.numel() == 1) and isinstance(
                self.ground_truth_output, float
        ):
            return abs(self.ground_truth_output - model_output.item()) >= 0.5
        else:
            return model_output.argmax() != self.ground_truth_output

    def _get_score(self, model_output, _):
        if (model_output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(model_output.item() - self.ground_truth_output)
        else:
            return 1 - model_output[self.ground_truth_output]

    def _call_model(self, attacked_text_list):
        acc_list = []
        logger = logging.getLogger(self.test_name)
        for text in attacked_text_list:
            logger.info("Current attacked text is: {}".format(text.text))
            acc = self.inference.predict(text.text)
            logger.info("Current acc:\t")
            logger.info(acc)
            acc_list.append(acc)
        return self._process_model_outputs(attacked_text_list, acc_list)
