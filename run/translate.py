import json
import torch
from torch.utils.data import DataLoader
from modelling.functional import TransformerModel
from dataset import MyDataset
from transformers import GPT2Tokenizer
import torch.nn.functional as F

def generate_translation(device, model, src_input, tokenizer, max_len=64):
    model.eval()

    bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    src_pad_idx = tokenizer.convert_tokens_to_ids("[PAD]")

    e_mask = (src_input.unsqueeze(0) != src_pad_idx).int()
    e_mask = e_mask.to(device)

    trg_input = torch.LongTensor([bos_token_id] + [src_pad_idx] * (max_len - 1))
    trg_input = trg_input.to(device)

    for i in range(max_len):
        d_mask = (trg_input.unsqueeze(0) != src_pad_idx).int()
        d_mask = d_mask.to(device)

        with torch.no_grad():
            output = model(src_input, trg_input, e_mask, d_mask)

        output = F.softmax(output, dim=-1)
        output = torch.argmax(output, dim=-1)
        word_id = output[0][i].item()

        trg_input[i+1] = word_id

        # Stop if the end of text token is generated
        if word_id == eos_token_id:
            break


    return trg_input

# Laden Sie das Modell und den Tokenizer
model = TransformerModel(
    vocab_size=50000,
    d_model=256,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=1024,
    dropout=0.2,
    max_len=64
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.load_state_dict(torch.load("/gpfs/project/flkar101/transformer_project/data/transformer_model.pth"))
model.to(device)

tokenizer = GPT2Tokenizer.from_pretrained("/gpfs/project/flkar101/transformer_project/gpt2_from_bpe")

# Laden Sie den Testdatensatz
test_data = torch.load("/gpfs/project/flkar101/transformer_project/data/test_dataset.pt")
test_dataset = MyDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Generieren Sie Übersetzungen für den Testdatensatz
translations = []
for i, batch in enumerate(test_loader):
    src_input, trg_output = torch.stack(batch['source']), torch.stack(batch['target_output'])
    src_input, trg_output = src_input.to(device), trg_output.to(device)

    translation = generate_translation(device, model, src_input, tokenizer)

    translation = translation[1:].tolist()

    src_sentence_text = tokenizer.decode(src_input, skip_special_tokens=True)
    correct_translation_text = tokenizer.decode(trg_output, skip_special_tokens=True)
    translation_text = tokenizer.decode(translation, skip_special_tokens=True)

    translations.append({'source': src_sentence_text, 'correct': correct_translation_text, 'generated': translation_text})

with open("/gpfs/project/flkar101/transformer_project/data/translations.json", "w") as f:
    json.dump(translations, f, indent=2)