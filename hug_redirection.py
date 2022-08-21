import os
import torch
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, DefaultDataCollator
import numpy as np

chkpt = 'deepset/roberta-base-squad2'
chkpt_dir = 'C:/Users/user/Documents' # local directory
max_answer_len = 30
qa_model = AutoModelForQuestionAnswering.from_pretrained(chkpt, cache_dir=chkpt_dir) # local_files_only=True, cache_dir=chkpt_dir,
tokenizer = AutoTokenizer.from_pretrained(chkpt, cache_dir=chkpt_dir) # cache directory 변경

context = '''
           Do you have any idea where to meet with counsler? I think the town hall will be find.
           Okay, then I'll book that room at 5o'clock. And I think we need to buy some snacks and alcohols for the meeting.
           There will be also many celebrities in the meeting so I am so excited to see them.
           Do you know who will come? I couldn't hear because the supervisor said it is classified.
           I only know that the famous actor, Kim BBam BBam who recently direct 'Margarita' will be there.
           '''
query = 'who will be at the party?'

def preprocess(query, context):
    inputs = tokenizer(query,
                       context,
                       max_length=100,
                       truncation='only_second',
                       stride=50,
                       return_overflowing_tokens=True,
                       return_offsets_mapping=True,
                       padding='max_length',
                       return_tensors='pt')
    sample_map = inputs.pop("overflow_to_sample_mapping")
    inputs["offset_mapping"] = [o if inputs.sequence_ids[k] == 1
                                   else None for k, o in enumerate(inputs["offset_mapping"])
    return inputs

qa_model.eval()
with torch.no_grad():
    inputs = preprocess(query, context)
    n_best = 20

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    outputs = qa_model(inputs['input_ids'])

    start_logits = outputs.start_logits.numpy()
    end_logits = outputs.end_logits.numpy()

    offsets = inputs["offset_mapping"]
    start_indexes = np.argsort(start_logits)[-1: -n_best - 1: -1].tolist()
    end_indexes = np.argsort(end_logits)[-1: -n_best - 1: -1].tolist()
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Skip answers that are not fully in the context
            if offsets[start_index] is None or offsets[end_index] is None:
                continue
            # Skip answers with a length that is either < 0 or > max_answer_length.
            if (end_index < start_index
                or end_index - start_index + 1 > max_answer_len):
                continue

            answers = {
                    "text": context[offsets[start_index][0]: offsets[end_index][1]],
                    "logit_score": start_logits[start_index] + end_logits[end_index],
                        }
    print(answers)