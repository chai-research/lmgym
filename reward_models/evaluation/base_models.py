from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class GPTJ():
    def __init__(self, params=None):
        model_name = "EleutherAI/gpt-j-6B"
        self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                truncation_side='left',
                padding_side='left'
                )
        self.tokenizer.pad_token_id = 50256
        self.model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().to(0)
        default_params = {
            'temperature': 0.72,
            'repetition_penalty': 1.13125,
            'max_new_tokens': 64,
            'top_p': 0.725,
            'top_k': 0,
            'do_sample': True,
            'eos_token_id': 198,
        }
        self.params = params or default_params

    def predict(self, text, best_of=4):
        inputs = [text] * best_of
        input_ids = self._get_tokenized_ids(inputs)
        outputs = self._generate_tokens(input_ids)
        responses = []
        for ins, outs in zip(inputs, outputs):
            decoded = self.tokenizer.decode(
                outs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            decoded = decoded[len(ins):]
            responses.append(decoded.rstrip())
        return responses

    def _get_tokenized_ids(self, inputs):
        input_ids = self.tokenizer(
            inputs,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True).to(0)
        return input_ids

    def _generate_tokens(self, input_ids):
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids['input_ids'],
                attention_mask=input_ids['attention_mask'],
                **self.params)
        return outputs
