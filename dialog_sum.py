import nlpcloud

def dialog_summary(dial_input):
    client = nlpcloud.Client("bart-large-samsum", "61f7921484faa96fdfbb4345dfbe050b017841bf", gpu=False, lang="en")
    return client.summarization(dial_input)