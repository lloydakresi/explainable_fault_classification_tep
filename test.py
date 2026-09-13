import argparse
import json
import time

import onnxruntime_genai as og
from common import (
    apply_chat_template,
    get_config,
    get_generator_params_args,
    get_guidance,
    get_guidance_args,
    get_user_prompt,
    get_search_options,
    register_ep,
    set_logger,
)

path = "explainable_fault_classification_tep\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4"


def main(model_path=path, ep="cpu"):
    config = get_config(model_path, ep)
    model = og.Model(config)

    tokenizer = og.Tokenizer(model)
    stream = tokenizer.create_stream()

    search_options = get_search_options(args)

    input_list = [
        {"role": "system", "content": args.system_prompt},
    ]
