import torch
import torch.nn as nn
from torch.nn import functional as F
import requests

batch_size = 64
block_size = 32
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda:0' if torch.cuda.is_available() else'cpu'
eval_iters = 200
n_embd = 384 #384/6=64
dropout = 0.2
n_head = 6
n_layer = 6

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
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1]for i in ix]).to(device)

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

class Head(nn.Module):
    #one head of self-attetion
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 缓存区，保存模型的各个参数
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        # dropout 可以存在所有有forward的地方,添加的主要原因是，网络已经很深了，为了避免过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)  # (B,T,16)
        q = self.query(x)  # (B,T,16)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))  # 对所有tril等于0，都会无穷大
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out

class MutiHeadAttention(nn.Module):
    # 多头注意力就是自注意机制并行
    def __init__(self, num_heads, head_size):
        super().__init__()
        # range 建立整数序列，Head(head_size)对每一个整数i都创建一个Head实例
        # ModuleList用于存储多个Module子模块
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim =-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    # 简单的线性层和非线性激活函数
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    # transformer block
    def __init__(self, n_embd,n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa =MutiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# 2-gram
class GPTLanguageMode(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.block = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # 嵌入层，将词汇索引转化为词嵌入向量，一个B对应一个C
        tok_emb = self.token_embedding_table(idx)  # (B,T,C) 这里面每一个都是一个字母
        pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # (T, C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B,T,vocab_size)

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
            # 不可以超过位置嵌入的范围
            idx_cond = idx[:, -block_size:]
            idx_cond = idx_cond.to(device)
            # prediction
            logits, loss = self(idx_cond)

            # focus on the last step only
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # dim = -1表示在最后一个维度进行
            # 从概率分布中抽取样本
            idx_next = torch.multinomial(probs, num_samples=1)
            # dim=1 表示在第二个维度（列）拼接
            idx = torch.cat((idx, idx_next), dim=1)
            idx = idx.to(device)
        return idx

model = GPTLanguageMode(vocab_size)
m = model.to(device)

# optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter== max_iters - 1:
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
idx = idx.to(device)
# 张量变成列表
print(decode(m.generate(idx, max_new_tokens = 400)[0].tolist()))