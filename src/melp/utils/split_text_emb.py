from transformers import AutoTokenizer, AutoModel
import torch

def split_text_after_emb(text, tokenizer, model):
    # Step 1: Tokenize the entire text
    # tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    inputs = tokenizer(text, return_tensors='pt')
    print(inputs["input_ids"])

    # Step 2: Embed the tokens
    # model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
    print(f'total embeddings shape: {embeddings.shape}')

    # Step 3: Split the embeddings by sentences
    # Split the text into sentences
    sentences = text.split('.')
    sentences = [sentence + '.' for sentence in sentences if sentence]

    # Tokenize each sentence to get the token lengths
    sentence_token_lengths = [len(tokenizer(sentence)['input_ids']) - 2 for sentence in sentences]  # Exclude [CLS] and [SEP]
    sentence_token_id = [tokenizer(sentence)['input_ids'] for sentence in sentences]
    # Split the embeddings based on token lengths
    start = 1  # Skip [CLS] token
    sentence_embeddings = []
    length_sum = 0
    for length in sentence_token_lengths:
        end = start + length
        sentence_embeddings.append(embeddings[start:end])
        start = end + 1  # Skip [SEP] token
        length_sum += length
    # # Print the results
    # for i, sen_embeddings in enumerate(sentence_embeddings):
    #     print(f"Sentence {i+1}: {sentences[i]}")
    #     print(f"Embeddings shape: {sen_embeddings.shape}")
    return sentence_embeddings