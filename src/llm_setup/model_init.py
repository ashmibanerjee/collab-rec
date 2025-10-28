import os
import torch
from typing import Optional
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from anthropic import AnthropicVertex
from openai import OpenAI
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import GenerativeModel as GenerativeModelPreview, Tool, grounding, \
    GenerationConfig
from mistralai_gcp import MistralGoogleCloud
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from src.llm_setup.vertexai_setup import initialize_vertexai_params, generate_acess_tokens
from importlib.metadata import PackageNotFoundError

load_dotenv()
OAI_API_KEY = os.getenv("OPENAI_API_KEY")
VERTEXAI_PROJECT = os.getenv("VERTEXAI_PROJECTID")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

HF_TEMPLATE = """{%- for message in messages %}{%- if message['role'] == 'user' %}{{- bos_token + '[INST] ' + 
message["content"].strip() + ' [/INST]' }}{%- elif message['role'] == 'system' %}{{- '<<SYS>>\\n' + message[
"content"].strip() + '\\n<</SYS>>\\n\\n' }}{%- elif message['role'] == 'assistant' %}{{- '[ASST] ' + message[
'content'] + ' [/ASST]' + eos_token }}{%- endif %}{%- endfor %}"""


def initialize_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


class LLMBaseClass:
    def __init__(self, model_id, location: Optional[str] = None) -> None:
        self.model_id = model_id
        self.location = location
        self.temperature = 0.5
        self.tokens = 1024
        self.terminators = None
        self.tokenizer = None
        try:
            self.bnb_config = initialize_bnb_config()
        except PackageNotFoundError:
            self.bnb_config = None
        self.model = self._initialize_model()

    def _initialize_model(self):
        mid = self.model_id[0].lower()

        if "gpt-4" in mid or "o4-mini" in mid:
            return OpenAI(api_key=OAI_API_KEY)
        if "deepseek" in mid:
            return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        if "claude" in mid:
            region = self.location or "europe-east5"
            return AnthropicVertex(region=region, project_id=VERTEXAI_PROJECT)
        if "gemini" in mid:
            return self._initialize_vertexai_model()
        if "llama" in mid:
            return self._initialize_vertexai_llama_maas()
        if "mistral-large" in mid:
            return self._initialize_vertexai_mistral()

        # Assume Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = HF_TEMPLATE
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=self.bnb_config,
        )
        model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        return model

    def _initialize_vertexai_model(self):
        mid = self.model_id[0]
        initialize_vertexai_params(location="us-central1" if "gemini-2.5" in mid or "flash" in mid else self.location)
        if "gemini-2.5" in mid:
            return genai.Client()
        elif "flash" in mid:
            return GenerativeModelPreview(mid)
        return GenerativeModel(mid)

    def _initialize_vertexai_llama_maas(self):
        initialize_vertexai_params(location="us-central1")
        credentials = generate_acess_tokens()
        endpoint = f"{self.location}-aiplatform.googleapis.com"
        base_url = f"https://{endpoint}/v1beta1/projects/{VERTEXAI_PROJECT}/locations/{self.location}/endpoints/openapi"
        return OpenAI(base_url=base_url, api_key=credentials.token)

    def _initialize_vertexai_mistral(self):
        credentials = generate_acess_tokens()
        return MistralGoogleCloud(
            access_token=credentials.token,
            region=self.location,
            project_id=VERTEXAI_PROJECT
        )

    def _generate_openai(self, messages, reasoning=False):
        if reasoning:
            completion = self.model.responses.create(
                model=self.model_id[0],
                input=messages,
                text={"format": {"type": "text"}},
                reasoning={"effort": "medium"}
            )
            return completion.output_text

        completion = self.model.chat.completions.create(
            model=self.model_id[0],
            messages=messages,
            temperature=0.6,
            top_p=0.9,
        )
        return completion.choices[0].message.content

    def _generate_vertexai(self, messages, is_grounded=False, web_search=False):
        mid = self.model_id[0]
        content = " ".join([msg["content"] for msg in messages])

        if "claude" in mid:
            response = self.model.messages.create(
                max_tokens=self.tokens,
                model=mid,
                system=messages[0]['content'],
                messages=[messages[1]],
            )
            return response.content[0].text
        if "llama" in mid or "mistral-large" in mid:
            method = self.model.chat.completions.create if "llama" in mid else self.model.chat.complete
            response = method(
                model=mid,
                messages=messages,
                max_tokens=self.tokens * 2,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        if "gemini-2.5" in mid:
            response = self.model.models.generate_content(
                model=mid,
                contents=content,
                config=GenerateContentConfig(thinking_config=ThinkingConfig(include_thoughts=True))
            )
            return response.text

        if is_grounded:
            tool = Tool.from_google_search_retrieval(
                grounding.GoogleSearchRetrieval()) if web_search else Tool.from_retrieval(
                grounding.Retrieval(
                    grounding.VertexAISearch(
                        datastore=os.environ["VERTEXAI_DATASTORE_ID"],
                        project=VERTEXAI_PROJECT,
                        location="global",
                    )
                )
            )
            return self.model.generate_content(
                content,
                tools=[tool],
                generation_config=GenerationConfig(
                    temperature=1.0,
                    max_output_tokens=8192,
                    top_p=0.95,
                ),
            )
        return self.model.generate_content(content).text

    def _generate_huggingface(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def generate(self, messages, is_grounded: Optional[bool] = False, web_search: Optional[bool] = False):
        mid = self.model_id[0].lower()

        if any(key in mid for key in ["gpt-4", "deepseek"]):
            return self._generate_openai(messages)
        if "o4-mini" in mid:
            return self._generate_openai(messages, reasoning=True)
        if any(key in mid for key in ["claude", "gemini", "llama", "mistral-large"]):
            return self._generate_vertexai(messages, is_grounded, web_search)
        return self._generate_huggingface(messages)
