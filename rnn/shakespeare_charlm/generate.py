import os
import torch
from model import CharRNN, CharLSTM

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def load_checkpoint(checkpoint_path, model_cls, device):
  # weights_only=False -- this checkpoint isn't just tensors, it also carries
  # plain-Python char2idx/idx2char dicts and int hyperparameters
  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
  model = model_cls(hidden_dim=checkpoint['hidden_dim'], embed_dim=checkpoint['embed_dim'], vocab_size=checkpoint['vocab_size'])
  model.load_state_dict(checkpoint['model_state_dict'])
  model = model.to(device)
  model.eval()
  return model, checkpoint['char2idx'], checkpoint['idx2char']


def generate(model, char2idx, idx2char, seed, length, device, temperature=1.0):
  is_lstm = isinstance(model, CharLSTM)
  h_prev = torch.zeros(1, model.hidden_dim, device=device)
  c_prev = torch.zeros(1, model.hidden_dim, device=device) if is_lstm else None

  def step(char_idx, h_prev, c_prev):
    # one character at a time -- T=1, so there's no batching win from
    # project_input here, plain forward() is simpler and exactly equivalent
    x_t = model.embedding(torch.tensor([[char_idx]], device=device)).squeeze(1)  # (1, embed_dim)
    if is_lstm:
      return model.lstm_cell.forward(x_t, h_prev, c_prev)  # h_t, c_t
    return model.rnn_cell.forward(x_t, h_prev), None  # h_t, None

  with torch.no_grad():
    # feed the seed through first, character by character, to build up h_prev
    # (and c_prev) -- everything except the LAST seed char also needs a step,
    # since that last char's step is what produces the first NEW prediction
    for ch in seed[:-1]:
      h_prev, c_prev = step(char2idx[ch], h_prev, c_prev)

    generated = list(seed)
    current_idx = char2idx[seed[-1]]

    for _ in range(length):
      h_prev, c_prev = step(current_idx, h_prev, c_prev)
      logits = model.classifier(h_prev)  # (1, vocab_size)
      # temperature: divide logits before softmax -- <1 sharpens the
      # distribution (more confident/repetitive), >1 flattens it (more random)
      probs = torch.softmax(logits / temperature, dim=-1)
      current_idx = torch.multinomial(probs, num_samples=1).item()
      generated.append(idx2char[current_idx])

  return ''.join(generated)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(42)

  seed_text = "ROMEO:"
  for label, model_cls, filename in [("RNN", CharRNN, "charrnn.pt"), ("LSTM", CharLSTM, "charlstm.pt")]:
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    model, char2idx, idx2char = load_checkpoint(checkpoint_path, model_cls, device)
    text = generate(model, char2idx, idx2char, seed=seed_text, length=400, device=device)
    print(f"--- {label} ---")
    print(text)
    print()
