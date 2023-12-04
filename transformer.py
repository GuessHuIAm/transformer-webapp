from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json

# Assuming 'emailspam' folder is in the current working directory
# Update the path if 'emailspam' is located elsewhere
model_path = 'emailspam'
tokenizer_path = 'emailspam'
common_tokens_path = 'emailspam/common_tokens.json'

# Load the model
model = BertForSequenceClassification.from_pretrained(model_path)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Load common tokens
with open(common_tokens_path, 'r') as file:
    common_tokens = json.load(file)

def create_modified_attention_mask(email, tokenizer, common_tokens, max_length=128):
    # Tokenize the email
    tokens = tokenizer.tokenize(email)

    # Create a modified attention mask
    modified_attention_mask = [0 if token not in common_tokens else 1 for token in tokens]

    # Pad or truncate the modified attention mask to the max_length
    while len(modified_attention_mask) < max_length:
        modified_attention_mask.append(0)  # Padding
    if len(modified_attention_mask) > max_length:
        modified_attention_mask = modified_attention_mask[:max_length]

    return modified_attention_mask

def transformer_api(text):
    # Tokenize the input text
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Create modified attention mask
    mod_attention_mask = create_modified_attention_mask(text, tokenizer, common_tokens)

    # Convert mod_attention_mask to a PyTorch tensor and reshape it
    mod_attention_mask_tensor = torch.tensor(mod_attention_mask, dtype=torch.long).unsqueeze(0)

    # Ensure the mask is the same size as the input IDs
    if mod_attention_mask_tensor.size(1) != encoded_dict['input_ids'].size(1):
        mod_attention_mask_tensor = mod_attention_mask_tensor[:, :encoded_dict['input_ids'].size(1)]

    # Set model to evaluation mode
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = model(encoded_dict['input_ids'], token_type_ids=None, attention_mask=mod_attention_mask_tensor)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).numpy()

    # Convert prediction to a readable format
    classification_label = 'Spam' if prediction == 1 else 'Ham'

    # Return both classification and modified attention mask
    return classification_label, mod_attention_mask

