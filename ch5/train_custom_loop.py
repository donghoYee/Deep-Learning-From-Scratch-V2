import matplotlib.pyplot as plt
import numpy as np
from dataset import ptb
from simple_rnnlm import SimpleRnnlm
from common.optimizer import SGD

batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = 0.1
max_epoch = 200

corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 4000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus)+1)

xs = corpus[:-1]
ts = corpus[1:]
data_size = len(xs)
print('size of corpus: %d, vocab : %d' %(corpus_size, vocab_size))

max_iters = data_size//(batch_size*time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

jump = (corpus_size - 1 )//batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = xs[(offset +time_idx) % data_size]
            time_idx += 1

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    ppl = np.exp(total_loss/loss_count)
    print('| epoch %d | perplexity %.2f' %(epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0,0

plt.plot(np.arange(max_epoch), ppl_list)
plt.show()

