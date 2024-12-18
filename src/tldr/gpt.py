# global interpreter
from typing import List
import asyncio
import numpy as np
from openai import OpenAI, AsyncOpenAI
import openai

api_key = ""
client = OpenAI(api_key=api_key)
aclient = AsyncOpenAI(api_key=api_key)

import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

forward_interpreter = None
backward_interpreter = None


class GPT:
    AVAILABLE_MODELS = [
        "text-davinci-003",
        "text-davinci-002",
        "code-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-turbo-preview",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]

    def __init__(self, model_name="text-davinci-003", **generation_options):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"model_name should be one of: {','.join(self.AVAILABLE_MODELS)}"
            )
        self.generation_options = generation_options
        self.engine = model_name

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.Timeout)
            | retry_if_exception_type(openai.APIError)
            | retry_if_exception_type(openai.APIConnectionError)
            | retry_if_exception_type(openai.RateLimitError)
        ),
    )
    async def aget_chat_completion_response(self, prompt, **kwargs):
        """
        prompting chatgpt via openai api
        now batching only works for completion, not on chat
        """
        if openai.api_type == "azure":
            try:
                response = await aclient.chat.completions.create(model=self.engine,
                messages=[{"role": "user", "content": prompt}],
                **kwargs)
            except client.BadRequestError as e:
                # Most likely a content filtering error from Azure.
                logging.warn(str(e))
                return str(e)
        else:
            response = await aclient.chat.completions.create(model=self.engine,
            messages=[{"role": "user", "content": prompt}],
            **kwargs)

        #if "content" not in response.choices[0].message:
        #    return ""

        output = response.choices[0].message.content.strip()
        return output

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.Timeout)
            | retry_if_exception_type(openai.APIError)
            | retry_if_exception_type(openai.APIConnectionError)
            | retry_if_exception_type(openai.RateLimitError)
        ),
    )
    def get_chat_completion_response(self, prompt, **kwargs):
        """
        prompting chatgpt via openai api
        now batching only works for completion, not on chat
        """
        if openai.api_type == "azure":
            try:
                response = client.chat.completions.create(deployment_id=self.engine,
                messages=[{"role": "user", "content": prompt}],
                **kwargs)
            except client.BadRequestError as e:
                # Most likely a content filtering error from Azure.
                logging.warn(str(e))
                return str(e)
        else:
            response = client.chat.completions.create(
                model=self.engine,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

        #if "content" not in response.choices[0].message:
        #    return ""

        output = response.choices[0].message.content.strip()
        return output

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.Timeout)
            | retry_if_exception_type(openai.APIError)
            | retry_if_exception_type(openai.APIConnectionError)
            | retry_if_exception_type(openai.RateLimitError)
        ),
    )
    def get_completion_response(
        self,
        prompt_batch,
        return_logprobs=False,
        raw_logprobs=False,
        top_logprobs=False,
        **kwargs,
    ):
        """
        prompting gpt-3 via openai api
        now batching only works for completion, not on chat
        """
        logging.debug(kwargs)

        try:
            response = client.completions.create(model=self.engine,
            prompt=prompt_batch,
            logprobs=top_logprobs or 1,
            **kwargs)
        except openai.BadRequestError as e:
            # Most likely a content filtering error from Azure.
            if "filtering" in str(e):
                logging.warn(str(e))
                # Process each element in the batch individually.
                response = {"choices": []}
                for prompt in prompt_batch:
                    try:
                        response.choices.append(
                            client.completions.create(model=self.engine,
                            prompt=prompt,
                            logprobs=top_logprobs or 1,
                            **kwargs)["choices"][0]
                        )
                    except client.BadRequestError as e:
                        response.choices.append(
                            {
                                "text": str(e),
                                "logprobs": {"token_logprobs": [0], "top_logprobs": [{}], "tokens": {}},
                            }
                        )
            else:
                raise e

        output = []
        nlls = []
        lengths = []
        for response in response.choices:
            output.append(response.text.strip())
            if raw_logprobs:
                nlls.append(response.logprobs.token_logprobs)
                lengths.append(response.logprobs.tokens)
            elif top_logprobs:
                nlls.append(response.logprobs.top_logprobs)
                lengths.append(response.logprobs.tokens)
            else:
                if "token_logprobs" in response.logprobs:
                    nlls.append(sum(response.logprobs.token_logprobs))
                    lengths.append(len(response.logprobs.token_logprobs))
                else:
                    nlls.append(-np.inf)
                    lengths.append(1)

        if return_logprobs:
            output = list(zip(output, nlls, lengths))
        return output

    async def gather_chat_response(self, inputs, **generation_options):
        outputs = await asyncio.gather(
            *[
                self.aget_chat_completion_response(_input, **generation_options)
                for _input in inputs
            ]
        )
        return outputs

    def generate(self, inputs, async_generation=True, **kwargs):
        if type(inputs) is not list:
            inputs = [inputs]

        kwargs.pop("output_space", None)
        generation_options = self.generation_options.copy()
        generation_options.update(**kwargs)

        if self.engine in ("gpt-3.5-turbo", "gpt-4", "gpt-4-32k", "gpt-4-turbo-preview", "gpt-4o", "gpt-4o-mini", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"):
            if "return_logprobs" in generation_options:
                del generation_options["return_logprobs"]

            if async_generation is True:
                # async
                outputs = asyncio.run(
                    self.gather_chat_response(inputs, **generation_options)
                )
            else:
                # call api one by one
                print(generation_options)
                outputs = [
                    self.get_chat_completion_response(_input, **generation_options)
                    for _input in inputs
                ]
        else:
            # devide to mini batches (max batch size = 20 according to openai)
            max_batch_size = 20
            input_length = len(inputs)
            num_batches = input_length // max_batch_size + (
                1 if input_length % max_batch_size > 0 else 0
            )
            outputs = []
            for i in range(num_batches):
                input_batch = inputs[max_batch_size * i : max_batch_size * (i + 1)]
                output_batch = self.get_completion_response(
                    input_batch, **generation_options
                )
                outputs = outputs + output_batch
        return outputs


def forward_instantiate(model_name="text-davinci-003", **generation_options):
    global forward_interpreter

    if forward_interpreter is None:
        forward_interpreter = GPT(model_name, **generation_options)
    else:
        print("Forward interpreter already instantiated.")
        pass


def backward_instantiate(model_name="text-davinci-003", **generation_options):
    global backward_interpreter

    if backward_interpreter is None:
        backward_interpreter = GPT(model_name, **generation_options)
    else:
        print("Backward interpreter already instantiated.")
        pass


def forward_evaluate(input: List[str], **kwargs):
    return forward_interpreter.generate(input, **kwargs)


def backward_evaluate(input: List[str], **kwargs):
    return backward_interpreter.generate(input, **kwargs)

if __name__ == '__main__':
    model = GPT(model_name='gpt-4o-mini')
    input = ['What is the name of the president?', 'Answer what is the color of the ocean?']
    out = model.generate(input, async_generation=True)
    print(out)