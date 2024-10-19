import json

from vllm import LLM as vLLM
from vllm.sampling_params import SamplingParams


def generate_prompt(labels, text):
    prompt = """
    Extract the entities for the following labels from the given text and provide the results in JSON format.
    - Entities must be extracted exactly as mentioned in the text.
    - Return each entity under its label without creating new labels.
    - Provide a list of entities for each label, ensuring that if no entities are found for a label, an empty list is returned.
    - Accuracy and relevance in your responses are key.

    labels:"""

    for label in labels:
        prompt += f"\n- {label}"

    prompt += """
    JSON Structure:
    {
    """

    for label in labels:
        prompt += f'"{label}": [],\n'

    prompt += "}\n\n"
    prompt += "\n\nTEXT:"
    prompt += text
    return prompt


# LLaMA3 and Mixtral
def llvm_ner(model_name, prompt):
    model_name = "gpt2"  # Use a publicly accessible model no login required
    sampling_params = SamplingParams(max_tokens=8192)
    llm = vLLM(
        model=model_name,
        tokenizer_mode="auto",
        load_format="auto",
    )

    if False:
        model_name = "mistralai/Mistral-NeMo-Instruct-2407"
        sampling_params = SamplingParams(max_tokens=8192)
        llm = vLLM(
            model=model_name,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
        )

    if False:
        model_name = "gpt-3.5-turbo"  # Use a publicly accessible model
        sampling_params = SamplingParams(max_tokens=8192)
        llm = vLLM(
            model=model_name,
            tokenizer_mode="openai",
            load_format="openai",
            # config_format="openai",
        )

    messages = [
        {
            'role': 'user',
            'content': prompt,
        },
    ]

    chat_template = """
    {% for message in messages %}
    {{ message.role }}: {{ message.content }}
    {% endfor %}
    """

    print(messages)
    res = llm.chat(
        messages=messages, sampling_params=sampling_params, chat_template=chat_template
    )
    print(res)
    return res


# Sample Text
labels = ['SKILLS', 'NAME', 'CERTIFICATE', 'HOBBIES', 'COMPANY', 'UNIVERSITY']

text = "I went to the University of California, Berkeley and graduated with a degree in Computer Science. I have a certificate in Data Science from Coursera. I have worked at Google for 5 years. My hobbies include playing the guitar and reading books. I am proficient in Python, Java, and C++."

# Results
# llama_result = ollama_ner('llama3', text)
prompt = generate_prompt(labels, text)
mixtral_result = llvm_ner('mixtral', prompt)

# print("LLaMA3 Result:", llama_result)
print("Mixtral Result:", mixtral_result)
