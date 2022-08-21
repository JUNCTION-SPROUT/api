import nlpcloud

NLP_API_KEY = "-"
context =  '''
           Do you have any idea where to meet with counsler? I think the town hall will be find.
           Okay, then I'll book that room at 5o'clock. And I think we need to buy some snacks and alcohols for the meeting.
           There will be also many celebrities in the meeting so I am so excited to see them.
           Do you know who will come? I couldn't hear because the supervisor said it is classified.
           I only know that the famous actor, Kim BBam BBam who recently direct 'Margarita' will be there.
           '''

def dialog_summary(dial_input):
    client = nlpcloud.Client("bart-large-samsum", NLP_API_KEY, gpu=False, lang="en")
    return client.summarization(dial_input)
print(dialog_summary(context))