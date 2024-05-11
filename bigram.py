import torch
import torch.nn as nn
from torch.nn import functional as F
import requests

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else'cpu'
eval_iters = 200

torch.manual_seed(1337)

# 准备数据集
url = "https://www.gutenberg.org/cache/epub/420/pg420.txt"
response = requests.get(url)
with open("input.txt", "wb") as f:
    f.write(response.content)

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

# set是做成一个集合，list成为列表，sorted排序
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 创建一个具有全部字母的integers
# 在第一个字典，键是字符，第二个数是字符
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
# lambda匿名函数，将字符串编码为整数
encode = lambda s: [stoi[c] for c in s]
# .join是连成字符串操作
decode = lambda l: ''.join([itos[i] for i in l])

# 对绿野仙踪全部的文本进行编码，然后包装到tensor中获得数据张量
data = torch.tensor(encode(text), dtype=torch.long)

# 划分数据集和验证集
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # 抓取一个batch的input x and target y
    data = train_data if split == 'train' else val_data
    # 当生成随机位置抓取数据的时候，生辰随机偏移量批量大小
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # stack 函数可以将数据按照给定索引'ix'切片堆叠成张量，作为行堆叠起来，成为4*8张量中的一行
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1]for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() #设置模式为eval
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 2-gram
class BigramLanguageMode(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # 嵌入层，将词汇索引转化为词嵌入向量，一个B对应一个C
        logits = self.token_embedding_table(idx)  # (B,T,C) 这里面每一个都是一个字母

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # 均为2维张量
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # 生成文本
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # prediction
            logits, loss = self(idx)
            # focus on the last step only
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # dim = -1表示在最后一个维度进行
            # 从概率分布中抽取样本
            idx_next = torch.multinomial(probs, num_samples=1)
            # dim=1 表示在第二个维度（列）拼接
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

model = BigramLanguageMode(vocab_size)
m = model.to(device)

# optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 抓取一个batch的数据
    xb, yb = get_batch('train')

    # loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1, 1),dtype = torch.long)
# 张量变成列表
print(decode(m.generate(idx, max_new_tokens = 400)[0].tolist()))