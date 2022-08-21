import os
from transformers import pipeline
import openai
import nlpcloud

question = 'who will be at the party?'
context =  '''
           Do you have any idea where to meet with counsler? I think the town hall will be find.
           Okay, then I'll book that room at 5o'clock. And I think we need to buy some snacks and alcohols for the meeting.
           There will be also many celebrities in the meeting so I am so excited to see them.
           Do you know who will come? I couldn't hear because the supervisor said it is classified.
           I only know that the famous actor, Kim BBam BBam who recently direct 'Margarita' will be there.
           '''

# Huggingface Bert
chkpt_dir = 'C:/Users/user/Documents'
def question_answering(query, context):
    chkpt = 'deepset/roberta-base-squad2' # 'deepset/roberta-base-squad2', 'bert-base-cased'
    QA = pipeline('question-answering', model=chkpt, cache_dir=chkpt_dir)

    return QA(question=query, context=context)['answer']

print(question_answering(question, context))

# GPT-3 API
OPENAI_API_KEY = '-'
openai.api_key = OPENAI_API_KEY

start_sequence = "\nA:"
restart_sequence = "\n\nQ: "

response = openai.Completion.create(
  model="text-davinci-002",
  prompt= context+"\n\nQ:"+question,
  temperature=0,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["\n"]
)
print(response)

# nlpcloud api
NLPCLOUD_API_KEY = "-"
client = nlpcloud.Client("finetuned-gpt-neox-20b", NLPCLOUD_API_KEY, gpu=False, lang="en")
out = client.question(question, context)
print(out)