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
1. CheckList
2. StressTest
3. PWWS
4. TextBugger
5. TextFooler
6. BAE
7. BERT-attack
8. CLARE
9. Deepwordbug
10. Alzantot
11. Faster-alzantot
12. IGA
13. LEAP
14. PSO
15. PRUTHI
16. Kuleshov
17. Input-reduction
18. ABS
19. ABFS
20. BASFuzz

## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](LICENSE) file. By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).


## Acknowledgement

This code is based on the [AORTA](https://github.com/lumos-xiao/ABS) framework.
