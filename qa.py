# Huggingface Bert
from transformers import pipeline

def question_answering(query, context):
    chkpt = 'deepset/roberta-base-squad2' # 'deepset/roberta-base-squad2' # 'bert-base-cased'
    QA = pipeline('question-answering', model=chkpt)

    return QA(question=query, context=context)['answer']

# print(question_answering('who will be at the party?',
#     '''
#                    Do you have any idea where to meet with counsler? I think the town hall will be find.
#                    Okay, then I'll book that room at 5o'clock. And I think we need to buy some snacks and alcohols for the meeting.
#                    There will be also many celebrities in the meeting so I am so excited to see them.
#                    Do you know who will come? I couldn't hear because the supervisor said it is classified.
#                    I only know that the famous actor, Kim BBam BBam who recently direct 'Margarita' will be there.
#                    '''))

# gpt-3 api
# import os
# import openai
#
# openai.api_key = ('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')
#
# start_sequence = "\nA:"
# restart_sequence = "\n\nQ: "
#
# response = openai.Completion.create(
#   model="text-davinci-002",
#   prompt="Do you have any idea where to meet with counsler? I think the town hall will be find. Okay, then I'll book that room at 5o'clock. And I think we need to buy some snacks and alcohols for the meeting.There will be also many celebrities in the meeting so I am so excited to see them. Do you know who will come? I couldn't hear because the supervisor said it is classified. I only know that the famous actor, Kim BBam BBam who recently direct 'Margarita' will be there.\n\nQ:who will be at the party?",
#   temperature=0,
#   max_tokens=100,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0,
#   stop=["\n"]
# )
# print(response)