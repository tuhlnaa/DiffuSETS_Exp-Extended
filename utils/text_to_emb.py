num = ['1st', '2nd', '3rd']

def prompt_propcess(text): 
    prompt_text = ''
    c = 0
    s = ''
    for ch in text:
        if ch == '|':
            # prompt_text += 'The ' + (num[c] if c <= 2 else str(c) + 'th') + ' diagnosis is {' + s + '}. '
            if c == 0:
                prompt_text += 'Most importantly, the 1st diagnosis is {' + s + '}.'
            else:
                prompt_text += 'As a supplementary condition, the ' + (num[c] if c <= 2 else str(c + 1) + 'th') + ' diagnosis is {' + s + '}.'
            c += 1
            s = ''
        else:
            s += ch
    if s != '':
        if c == 0:
            prompt_text += 'Most importantly, the 1st diagnosis is {' + s + '}.'
        else:
            prompt_text += 'As a supplementary condition, the ' + (num[c] if c <= 2 else str(c + 1) + 'th') + ' diagnosis is {' + s + '}.'
        c += 1
        s = ''

    # print(prompt_text)
    return prompt_text 

def get_text_embedding(text_batch, text_embed_table): 
    # text_batch -> (B, 1536) 
    text_embed = [] 
    for text in text_batch:
        prompt_text = prompt_propcess(text) 
        if len(text_embed_table.loc[text_embed_table['text'] == prompt_text, 'embed']) > 0:
            embed = text_embed_table.loc[text_embed_table['text'] == prompt_text, 'embed'].values[0]
        else:
            print(1)
            embed = text_embed_table.iloc[-1]['embed']
        embed = eval(embed)
        text_embed.append(embed)

    return text_embed 
