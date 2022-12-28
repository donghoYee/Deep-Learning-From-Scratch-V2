from better_rnnlm_gen import BetterRnnlmGen
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = BetterRnnlmGen()

model.load_params("ch6/BetterRnnlm.pkl")

start_word = "i"
start_id = word_to_id[start_word]
skip_words = ["N", "<unk>", "$"]
skip_ids = [word_to_id[i] for i in skip_words]

word_ids = model.generate(start_id, skip_ids)
txt = " ".join([id_to_word[i] for i in word_ids])
txt = txt.replace(" <eos>", ".\n")
print(txt)
