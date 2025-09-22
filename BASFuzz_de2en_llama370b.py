from inference import Inference
import textattack
from textattack import Attack
from textattack.goal_functions import MinimizeBleu
from types import SimpleNamespace
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from datasets import load_dataset
from textattack.transformations import WordSwapGEMMABOT
from textattack.search_methods import BeamAnnealingSearch

test_name = 'de-en_BAS_Llama3-70B'
text_name = test_name + '.txt'
csv_name = test_name + '.csv'
args = SimpleNamespace(text="some text")
wmt = load_dataset('./wmt16', "de-en")
args.model = 'Llama3-70B'
args.test_name = test_name
args.log_name = test_name + '.log'

inference_model = Inference(args)
goal_function = MinimizeBleu(inference=inference_model,
                             logger=args.test_name,
                             model_wrapper=None,
                             target_bleu=0.2)
transformation = WordSwapGEMMABOT()
constraints = [RepeatModification(), StopwordModification()]
search_method = BeamAnnealingSearch()
prompt = "Please translate the following German text into English. The output should contain only the translated English text, with no additional explanations or content:"
attack = Attack(goal_function, constraints, transformation, search_method)
test_data = wmt['test']
example_ori = []
n = 1000
for i in range(min(n, len(test_data))):
    src_text = prompt + test_data[i]['translation']['de']
    ref_text = test_data[i]['translation']['en']
    if len(src_text) >= 1000:
        src_text = src_text[:1000]
    example_ori.append((src_text, ref_text))

dataset = textattack.datasets.Dataset(example_ori)
attack_args = textattack.AttackArgs(num_examples=1000, log_to_txt=text_name, disable_stdout=True, shuffle=True,
                                    random_seed=42, log_to_csv=csv_name, parallel=False)
attacker = textattack.Attacker(attack, dataset, attack_args)
