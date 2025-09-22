import re
import numpy as np
import torch
from modelscope import Model
from modelscope.models.nlp.llama2 import Llama2Tokenizer, Llama2Config
import transformers
from transformers import BitsAndBytesConfig, pipeline
from modelscope import AutoModelForCausalLM, AutoTokenizer


GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
PIPE = None

def get_formatted_input(messages):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(
        ["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in
         messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + conversation

    return formatted_input


def load_model(model_name):
    """
    Load the model and tokenizer.
    """
    global GLOBAL_MODEL, GLOBAL_TOKENIZER, PIPE
    # Add loading logic for different models here
    if model_name == "Llama-2-70b-chat-ms":
        model_dir = "Llama-2-70b-chat-ms"
        model_config = Llama2Config.from_pretrained(model_dir)
        model_config.pretraining_tp = 1
        tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True)
        model = Model.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            config=model_config,
            quantization_config=quantization_config,
            device_map='auto'
        )
    elif model_name == "Mistral-7B-Instruct-v0.3":
        model_dir = "Mistral-7B-Instruct-v0___3"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        PIPE = pipe
    elif model_name == "phi4":
        model_dir = "phi-4"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        PIPE = pipe
    elif model_name == "Yi-1.5-34B":
        model_dir = "/media/xmx/data/Yi-1.5-34B-Chat/01ai/Yi-1___5-34B-Chat-16K"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16,
        ).eval()
    elif model_name == "Llama3-70B":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_4bit=False
        )
        model_dir = 'Llama3-70B'
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16,
                                                     quantization_config=quantization_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    elif model_name == "internlm2_5-20b":
        model_dir = "internlm2_5-20b-chat"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True,
                                                     torch_dtype=torch.float16).cuda()
        model = model.eval()

    GLOBAL_MODEL = model
    GLOBAL_TOKENIZER = tokenizer


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

"""
This clss implements the inference of the model (including create the model).
"""
class Inference(object):
    def __init__(self, args):
        self.args = args
        self.model = args.model
        load_model(self.model)

    def predict(self, attacked_text=None):
        if self.model in ['llama_2_70b', 'bloomz_7b1', 'llama_2_7b', 'llama_2_13b', 'qianfan_bloomz_7b_compressed',
                          'chatglm2_6b_32k', 'aquilachat_7b']:
            results = self.predict_by_api(self.model, attacked_text)
        else:
            results = self.predict_by_local_inference(self.model, attacked_text)
        return results

    def predict_by_generation(self, model, attacked_text):
        global GLOBAL_MODEL, GLOBAL_TOKENIZER, PIPE
        if model == "Llama-2-70b-chat-ms":
            system = "Please only output the label and the confidence score to three decimal places, in the format \"[negative]+[confidence score for negative],[neutral]+[confidence score for neutral],[positive]+[confidence score for positive]\", and nothing else. Don't write explanations nor line breaks in your replies."
            inputs = {'text': attacked_text, 'system': system, 'max_length': 512}
            output = GLOBAL_MODEL.chat(inputs, GLOBAL_TOKENIZER)
            response = output['response']
        elif model == "internlm2-chat-20b" or model == "internlm2_5-20b":
            response, history = GLOBAL_MODEL.chat(GLOBAL_TOKENIZER, attacked_text, history=[])
        elif model == "Mistral-7B-Instruct-v0.3":
            generation_args = {
                "max_new_tokens": 500,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }
            output = PIPE(attacked_text, **generation_args)
            response = output[0]['generated_text']
        elif model == "Phi-3-medium":
            generation_args = {
                "max_new_tokens": 500,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }
            output = PIPE(attacked_text, **generation_args)
            response = output[0]['generated_text']
        elif model == "Yi-1.5-34B":
            generation_args = {"max_new_tokens": 500}
            messages = [
                {"role": "user", "content": attacked_text}
            ]
            input_ids = GLOBAL_TOKENIZER.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
            output_ids = GLOBAL_MODEL.generate(input_ids.to('cuda'), eos_token_id=GLOBAL_TOKENIZER.eos_token_id,
                                               **generation_args)
            response = GLOBAL_TOKENIZER.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        elif model == "Llama3-70B":
            messages = [
                {"role": "user", "content": attacked_text}
            ]
            formatted_input = get_formatted_input(messages)
            tokenized_prompt = GLOBAL_TOKENIZER(GLOBAL_TOKENIZER.bos_token + formatted_input, return_tensors="pt").to(
                GLOBAL_MODEL.device)
            terminators = [
                GLOBAL_TOKENIZER.eos_token_id,
                GLOBAL_TOKENIZER.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = GLOBAL_MODEL.generate(input_ids=tokenized_prompt.input_ids,
                                            attention_mask=tokenized_prompt.attention_mask,
                                            max_new_tokens=500, eos_token_id=terminators)
            response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
            response = GLOBAL_TOKENIZER.decode(response, skip_special_tokens=True)
        elif model == "phi4":
            messages = [
                {"role": "system",
                 "content": "The output should contain only the translated English text, with no additional explanations or content."},
                {"role": "user", "content": attacked_text},
            ]
            outputs = PIPE(messages, max_new_tokens=500)
            response = outputs[0]["generated_text"][-1]["content"]
        else:
            raise ValueError(f"Unsupported model: {model}. Please check the model name and try again.")
        return response

    def predict_by_local_inference(self, model, attacked_text):
        raw_pred = self.predict_by_generation(model, attacked_text)
        while raw_pred is None:
            raw_pred = self.predict_by_generation(model, attacked_text)
        pred = self.process_raw_predict(model, raw_pred)
        return pred

    def process_raw_predict(self, model, raw_pred):
        src = ""
        result_array = ''
        if model == "Llama-2-13b-chat-ms":
            text_src = raw_pred
            start_index = text_src.find(":")
            if start_index != -1:
                src = text_src[start_index + 1:].strip()
                src = src.replace("\n", " ")
                return src
        elif model == "Llama-2-70b-chat-ms" or model == "internlm2-chat-20b" or model == "internlm2_5-20b":
            text_src = raw_pred
            src = process_text_mistral(text_src)
            return src
        elif model == "Mistral-7B-Instruct-v0.2":
            text = raw_pred
            text = text.split('\n')[0]
            text = re.sub(r'(?<!\d)(\.\d+)', r'0\1', text)
            text = text.replace("\n", " ")
            words = re.findall(r'[a-zA-Z]+', text)
            word_count = len(words)
            if word_count == 0:
                result_array = [0.000, 0.000, 0.000]
                return result_array
            if word_count > 4:
                result_array = [0.000, 0.000, 0.000]
                return result_array
            matches = re.findall(r'(?:\d+\.\d+,){2}\d+\.\d+', text)
            if matches:
                result_array = [0.000, 0.000, 0.000]
                return result_array
            text = remove_before_first_bracket(text)
            text = text.replace("]", "] ")
            text = text.strip().lower().replace("<pad>", "").replace("</s>", "")
            text = text.replace("[", "").replace("]", "")
            text = text.replace("+", " ").replace("-", " ")
            text = text.replace(":", " ").replace(",", " ")
            text = text.replace("\"", " ").replace("\'", " ")
            text = text.replace("\\", " ").replace("=", " ")
            text = text.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
            matches = re.findall(r'(negative|neutral|positive)\s+(\d+\.\d+)', text)
            category_dict = {match[0]: float(match[1]) for match in matches}
            categories = ['negative', 'neutral', 'positive']
            for i, category in enumerate(categories):
                result_array[i] = category_dict.get(category, 0.001)
            score_sum = sum(result_array)
            if 0.9 <= score_sum <= 1.1:
                return result_array
            else:
                if np.array_equal(result_array, [0, 0, 0]):
                    result_array = [0.333, 0.333, 0.333]
                result_array = softmax(result_array)
                score_sum = sum(result_array)
                if 0.9 <= score_sum <= 1.1:
                    return result_array
                else:
                    print("error: score_sum not in [0.9, 1.1] after softmax")
                    print(f"predict ori: {raw_pred}")
                    result_array = [0.000, 0.000, 0.000]
                    return result_array
        elif model == "Mixtral-8x7B-Instruct-v0.1":
            text = raw_pred
            text = text.split('\n')
            text = [line for line in text if not line.startswith("I want you")]
            text = '\n'.join(text)
            text = text.replace("\n", " ")
            contains_keywords = any(word in text for word in ["negative", "neutral", "positive"])
            contains_digit = any(char.isdigit() for char in text)
            if not (contains_keywords and contains_digit):
                result_array = [0.000, 0.000, 0.000]
                return result_array
            text = remove_before_first_bracket(text)
            text = text.replace("]", "] ")
            text = text.strip().lower().replace("<pad>", "").replace("</s>", "")
            text = text.replace("[", "").replace("]", "")
            text = text.replace("+", " ").replace("-", " ")
            text = text.replace(":", " ").replace(",", " ")
            text = text.replace("\"", " ").replace("\'", " ")
            text = text.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
            matches = re.findall(r'(negative|neutral|positive)\s+(\d+\.\d+)', text)
            category_dict = {match[0]: float(match[1]) for match in matches}
            categories = ['negative', 'neutral', 'positive']
            for i, category in enumerate(categories):
                result_array[i] = category_dict.get(category, 0.001)
            score_sum = sum(result_array)
            if 0.9 <= score_sum <= 1.1:
                return result_array
            else:
                if np.array_equal(result_array, [0, 0, 0]):
                    result_array = [0.333, 0.333, 0.333]
                result_array = softmax(result_array)
                score_sum = sum(result_array)
                if 0.9 <= score_sum <= 1.1:
                    return result_array
                else:
                    print("error: score_sum not in [0.9, 1.1] after softmax")
                    print(f"predict ori: {raw_pred}")
                    result_array = [0.000, 0.000, 0.000]
                    return result_array
        elif model == "Yi-34B-Chat" or model == "Yi-1.5-34B" or model == "Llama3-70B":
            text_src = raw_pred
            src = process_text_mistral(text_src)
            return src
        elif model == "Mistral-7B-Instruct-v0.3":
            text_src = raw_pred
            src = process_text_mistral(text_src)
            return src
        elif model == "Phi-3-medium":
            text_src = raw_pred
            src = process_text_phi(text_src)
            return src
        elif model == "phi4":
            text_src = raw_pred
            src = process_text_mistral(text_src)
            return src
        else:
            raise ValueError(f"Unsupported model: {model}. Please check the model name and try again.")


def remove_before_first_bracket(text):
    parts = text.split("[", 1)
    if len(parts) > 1:
        return "[" + parts[1]
    else:
        return text


def keep_line_with_first_bracket(text):
    # Split the text into lines and keep only the line with the first bracket
    lines = text.split("\n")
    for line in lines:
        if "[" in line:
            return line
    return ""


def process_text_phi(input_text):
    if '#' not in input_text:
        return ''
    first_hash_index = input_text.index('#')
    line_end = input_text.find('\n', first_hash_index)
    if line_end == -1:
        line_end = len(input_text)
    input_text = input_text[line_end + 1:]
    input_text = input_text.lstrip()
    if '\n' in input_text:
        processed_text = input_text.split('\n', 1)[0]
    else:
        processed_text = input_text
    return processed_text


def process_text_mistral(input_text):
    lines = input_text.split('\n')
    word_pattern = re.compile(r'\w+')
    for line in reversed(lines):
        if word_pattern.search(line):
            return line.strip()
    return ''


def process_text_fi_enllama(input_text):
    lines = input_text.split('\n')
    word_pattern = re.compile(r'\w+')
    quote_pattern = re.compile(r'^["\'](.*?)["\']$')
    for line in reversed(lines):
        if word_pattern.search(line):
            stripped_line = line.strip()
            match = quote_pattern.match(stripped_line)
            if match:
                return match.group(1)
            return stripped_line

    return ''
