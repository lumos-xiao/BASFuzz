# BASFuzz

This is the official source code for "BASFuzz: Towards Robustness Evaluation of LLM-based NLP Software via Automated Fuzz Testing". The workflow of BASFuzz is illustrated in the following figure:

![The Workflow of BASFuzz](/images/BASFuzz_workflow.png "The Workflow of BASFuzz")

## An example of robustness flaws in LLM-based NLP software
In a contract translation scenario, subtly perturbed input text (red) misleads the software, causing it to mistranslate a clause expressing permission to terminate the contract into an obligation to terminate.
![An example of robustness flaws in LLM-based NLP software](/images/BASFuzz_fig1.png "An example of robustness flaws in LLM-based NLP software")

## Repo structure
- `datasets`: define the dataset object used for carrying out tests
- `goal_functions`: determine if the testing method generates successful test cases
- `search_methods`: explore the space of transformations and try to locate a successful perturbation
- `transformations`: transform the input text, e.g. synonym replacement
- `constraints`: determine whether or not a given transformation is valid

The most important files in this project are as follows:
- `goal_functions/text/minimize_bleu.py`: quantify the goal of testing LLM-based NLP software in machine translation task
- `goal_functions/classification/untargeted_llm_classification.py`: quantify the goal of testing LLM-based NLP software in text classification task
- `search_methods/bas_word_swap_wir.py`: search test cases based on beam-annealing search
- `inference.py`: drive threat models to do inference and process outputs
- `BASFuzz_de2en_llama370b.py`: test Llama-3-70B-Instruct on the WMT16 dataset via BASFuzz

## Datesets
| Dataset               | Download Link                                                                 |
|-----------------------|-------------------------------------------------------------------------------|
| WMT16                 | [https://huggingface.co/datasets/wmt/wmt16](https://huggingface.co/datasets/wmt/wmt16)                           |
| Financial Phrasebank  | [https://huggingface.co/datasets/financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) |
| AG's News             | [https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz)             |
| MR                   | [https://huggingface.co/datasets/rotten_tomatoes](https://huggingface.co/datasets/rotten_tomatoes)                        |


## Threat models
| Threat model              | Download Link                                                                     |
|---------------------------|-----------------------------------------------------------------------------------|
| Mistral-7B-Instruct-v0.3  | [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| Phi-4                     | [https://huggingface.co/microsoft/phi-4](https://huggingface.co/microsoft/phi-4)                                  |
| InternLM2.5-20B-Chat      | [https://huggingface.co/internlm/internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)     |
| Yi-1.5-34B-Chat           | [https://huggingface.co/01-ai/Yi-1.5-34B-Chat](https://huggingface.co/01-ai/Yi-1.5-34B-Chat)                   |
| Llama-3-70B-Instruct      | [https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) |

## Dependencies
The code is tested with:

- bert-score>=0.3.5
- autocorrect==2.6.1
- accelerate==0.25.0
- datasets==2.15.0
- nltk==3.8.1
- openai==1.3.7
- sentencepiece==0.1.99
- tokenizers==0.15.0
- torch==2.1.1
- tqdm==4.66.1
- transformers==4.38.0
- Pillow==10.3.0
- transformers_stream_generator==0.0.5
- matplotlib==3.8.3
- tiktoken==0.6.0

## How to Run:
Follow these steps to run the attack from the library:

1. Fork this repository

2. Run the following command to install it.

   ```bash
   $ pip install -e . ".[dev]"
   
3. Run the following command to test Llama-3-70B-Instruct on the WMT16 dataset via BASFuzz.

   ```bash
   $ python BASFuzz_de2en_llama370b.py

Take a look at the `Models` directory in [Hugging Face](https://huggingface.co/models) to run the test across any threat model.

## Available test methods
| Method | Paper |
|:-----|:----:|
| CheckList | [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://www.ijcai.org/proceedings/2021/659) |
| StressTest | [Stress Test Evaluation for Natural Language Inference](https://aclanthology.org/C18-1198/) |
| PWWS | [Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency](https://www.aclweb.org/anthology/P19-1103/) |
| TextBugger | [TextBugger: Generating Adversarial Text Against Real-world Applications](https://www.ndss-symposium.org/ndss-paper/textbugger-generating-adversarial-text-against-real-world-applications/) |
| TextFooler | [Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment](https://ojs.aaai.org/index.php/AAAI/article/view/6311) |
| BAE | [BAE: BERT-based Adversarial Examples for Text Classification](https://aclanthology.org/2020.emnlp-main.498/) |
| BERT-attack | [BERT-ATTACK: Adversarial Attack Against BERT Using BERT](https://aclanthology.org/2020.emnlp-main.500/) |
| CLARE | [Contextualized Perturbation for Textual Adversarial Attack](https://aclanthology.org/2021.naacl-main.400/) |
| DeepWordBug | [Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers](https://ieeexplore.ieee.org/document/8424632) |
| Alzantot | [Generating Natural Language Adversarial Examples](https://aclanthology.org/D18-1316/) |
| Faster-Alzantot | [Certified Robustness to Adversarial Word Substitutions](https://aclanthology.org/D19-1423/) |
| IGA | [Natural Language Adversarial Defense through Synonym Encoding](https://proceedings.mlr.press/v161/wang21a.html) |
| LEAP | [LEAP: Efficient and Automated Test Method for NLP Software](https://ieeexplore.ieee.org/abstract/document/10298415/) |
| PSO | [Word-level Textual Adversarial Attacking as Combinatorial Optimization](https://www.aclweb.org/anthology/2020.acl-main.540/) |
| Pruthi | [Combating Adversarial Misspellings with Robust Word Recognition](https://aclanthology.org/P19-1561/) |
| Kuleshov | [Adversarial Examples for Natural Language Classification Problems](https://openreview.net/pdf?id=r1QZ3zbAZ) |
| Input-reduction | [Pathologies of Neural Models Make Interpretations Difficult](https://pdfs.semanticscholar.org/18eb/c6dfa3ed6096e6200cc74b8d29c75c13706d.pdf) |
| ABS | [Automated Robustness Testing for LLM-based Natural Language Processing Software](https://arxiv.org/abs/2412.21016) |
| ABFS | [ABFS: Natural Robustness Testing for LLM-based NLP Software](https://arxiv.org/pdf/2503.01319) |
| BASFuzz | [BASFuzz: Towards Robustness Evaluation of LLM-based NLP Software via Automated Fuzz Testing](https://arxiv.org/pdf/2509.17335) |

## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](LICENSE) file. By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).


## Acknowledgement

This code is based on the [AORTA](https://github.com/lumos-xiao/ABS) framework.
