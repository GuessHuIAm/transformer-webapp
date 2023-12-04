from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the model and tokenizer
model_path = '/emailspam'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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

sample_spam_email = """
Subject: Urgent Account Verification Required!

Dear Valued Customer,

We've noticed some unusual activity on your account and believe it may have been accessed by an unauthorized third party. For your security, we've temporarily locked your account.

To restore access, please verify your account information by clicking the link below. This is a mandatory security measure to ensure your account's integrity.

[Secure Verification Link]

Please note, failure to complete this verification within 24 hours of receiving this email will result in your account being permanently closed.

Thank you for your prompt attention to this matter.

Best regards,
[Your Bank's Name] Security Team
"""

# Example call to transformer_api with a sample spam email
classification_result, attention_mask = transformer_api(sample_spam_email)

print("Classification:", classification_result)
print("Modified Attention Mask:", attention_mask)