# _*_coding:utf-8 _*_
# @Time : 2022/9/4 
# @Author : Lin
# 还是改回预测位置，不再预测下一个结点的概率了！
# blockM+linkM_position ，在全家福的基础上去掉node_type
from random import shuffle
from operator import index
import pickle
import torch
from matplotlib import pyplot as plot
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import math
import torch.optim as optim
import os
import random
import time

# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
device = 'cuda'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device_ids = [0, 1] 	# id为0和1的两块显卡


n_iter = 45  # default 2
n_doc=60
batch_size = n_iter  # defaut 1
d_Q = 64  # defualt 64
d_V = 64  # defualt 64
num_of_node = 45  # defualt  20
d_feature = 3  #
d_node_type=24 #
d_model = 256  # defualt 512
n_layers = 6  # default 6
n_head = 1  # defualt 8
d_ff = 256  # default 64
epochs = 10000  # default 10000

def position_normal(position):
    position[:,0]=position[:,0]/max(position[:,0])
    position[:, 1] = position[:,1] / max(position[:, 1])
    position[:, 2] = position[:,2] / max(position[:, 2])
    return position

def sep_train_test(filename): # filename='short_to_long.txt'
    f=open(filename,'r')
    a=f.read()
    short_to_long=np.mat(a)
    stl_list=[]
    for i in range (n_doc):    # 这里更改读取数据的长度可以限制去掉长序列
        stl_list.append(short_to_long[0,i])
    print(stl_list)
    f.close()
    shuffle(stl_list)
    txt_file = open('saved_shuffle/p2p_blockM_linkM_position/shuffle.txt', "a", encoding="utf-8")
    txt_file.write(str(stl_list))
    return stl_list # 这里的stl_list已经是乱序了！

def shuffle_read_train(filename): #filename='saved_shuffle/p2p_blockM_position_type/shuffle.txt'
    f=open(filename, 'r')
    a=f.read().split(',')
    # print(type(a))
    # print(a)
    sh_list=[]
    for i in a:
        sh_list.append(int(i))
    return sh_list 

def read_data(filename):  #filename = 'whole_data.data'
    f = open(filename, 'rb')
    data=pickle.load(f)
    linkM=data['linkM']
    position=data['position']
    position=position_normal(position)
    node_type=data['node_type']    # 这个需要转化成输出形式是和enc_input 是一样的
    order=data['order']
    blockM=data['blockM']    # blockM  先进行转置 维度从 n_of_node*n_of_node*6_direction --> 6_direction*n_of_node*n_of_node  
    blockM=torch.transpose(blockM, 0,2)
    output_pos=position[order.long()]  # 这个是benchmark
    node_type=node_type[order.long()]
    return linkM, blockM,position,node_type, output_pos, order


# 首先构建数据样式
def make_data(n_iter):  # 这里的n_iter代表一共有多少个样本， 样本需要从文件中读取出来，这里先随机生成
    filename_prefix = 'data/LEGO'
    filename_suffix = '.data'
    linkM = torch.empty(0, num_of_node, num_of_node)   # 这里是linkM
    position_enc_input = torch.empty(0, num_of_node, d_feature)
    position_dec_input = torch.empty(0, num_of_node, d_feature)
    position_dec_output = torch.empty(0, num_of_node, d_feature)
    node_type_enc_input=torch.empty(0,num_of_node,d_node_type)
    blockM =torch.empty(0, 6, num_of_node,num_of_node)   # blockM 这里变成四维的，第一个维度是batch_size
    # stl_list=sep_train_test('short_to_long.txt')
    stl_list=shuffle_read_train('shuffle.txt')
    order=torch.empty(0,num_of_node).long()
    for i in range(n_iter):
        filename = filename_prefix + str(stl_list[i]-1) + filename_suffix        
        linkM_item,blockM_item, position_item, node_type_item, output_pos_item, order_item = read_data(filename)
        padding_len = num_of_node - linkM_item.shape[0]
        # 上面的item.shape[0]代表的是实际上有多少个结点
        order_added = np.ones([1,num_of_node])*-1
        order_added[0,0:len(order_item)] = order_item
        order_added=torch.tensor(order_added)
        order_added = order_added
        order=torch.cat([order,order_added.long()],dim=0)
        # 给link 增加padding
        linkM_right = torch.zeros([linkM_item.shape[0], padding_len])
        linkM_item = torch.cat([linkM_item, linkM_right], dim=1)
        linkM_bottom = torch.zeros([padding_len, linkM_item.shape[1]])
        linkM_item = torch.cat([linkM_item, linkM_bottom], dim=0)
        linkM = torch.cat([linkM, linkM_item.unsqueeze(0)], dim=0)   # 这里是在增加batch_size
        # 给 blockM  加上padding
        blockM_right=torch.zeros([6,blockM_item.shape[1],padding_len])
        blockM_item=torch.cat([blockM_item, blockM_right],axis=2)
        blockM_bottom=torch.zeros([6, padding_len, blockM_item.shape[2]])
        blockM_item=torch.cat([blockM_item,blockM_bottom],axis=1)
        blockM=torch.cat([blockM,blockM_item.unsqueeze(0)],dim=0)  # 这里是在增加batch_size
        # 给enc_input增加padding
        position_added = torch.cat([position_item, torch.ones(padding_len, d_feature) * -1], dim=0)  # 添加padding
        position_enc_input = torch.cat([position_enc_input, position_added.float().unsqueeze(0)], dim=0)  # 增加batch
        # 给node_type_enc_input 增加padding
        node_type_add=torch.cat([node_type_item,torch.ones(padding_len, d_node_type)*-1],dim=0) 
        node_type_enc_input=torch.cat([node_type_enc_input, node_type_add.float().unsqueeze(0)],dim=0) 
        # 给dec_input 增加开始的token 以及padding
        position_added = torch.cat([torch.ones(1, d_feature) * -1, output_pos_item], dim=0)
        position_added = torch.cat([position_added, torch.ones(padding_len - 1, d_feature) * -1], dim=0)
        position_dec_input = torch.cat([position_dec_input, position_added.float().unsqueeze(0)],
                                       dim=0)  # 增加batch
        # 给position_dec_output 添加结束的token 以及padding
        # 这里输出的position的顺序改变了！用的是output_pos_item,现在它是ground truth
        position_added = torch.cat([output_pos_item, torch.ones(padding_len, d_feature) * -1], dim=0)  # todo 终止符是否有必要？
        position_dec_output = torch.cat([position_dec_output, position_added.float().unsqueeze(0)],
                                        dim=0)  # 增加batch

    return linkM, blockM, position_enc_input, position_dec_input, position_dec_output,node_type_enc_input, order


# 这里建立一个data_loader
class Mydataset(Data.Dataset):
    def __init__(self,linkM,blockM, position_graph, position_seq,position_real,node_type,order):
        super(Mydataset, self).__init__()
        self.linkM=linkM.to(device)
        self.blockM=blockM.to(device)
        self.position_graph=position_graph.to(device)
        self.position_seq=position_seq.to(device)
        self.position_real=position_real.to(device)
        self.node_type=node_type.to(device)
        self.order=order.to(device)
    def __len__(self):
        return self.linkM.shape[0]
    def __getitem__(self, idx):
        return self.linkM[idx].to(device),self.blockM[idx].to(device), self.position_graph[idx].to(device),self.position_seq[idx].to(device), \
        self.position_real[idx].to(device), self.node_type[idx].to(device), self.order[idx].to(device)

# 这里建立一个从position到embedding的全连接层
class position2embedding(nn.Module):
    def __init__(self):
        super(position2embedding, self).__init__()
        self.fc = nn.Linear(d_feature, d_model)

    def forward(self, position):
        embedding = self.fc(position)
        return embedding.to(device)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        score = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_Q)
        score.masked_fill_(attn_mask.bool(), -1e9)  # 利用attn_mask 对score响应位置置为接近0的值
        # plot.matshow(score[0,0,:,:].squeeze(0).squeeze(0).detach().numpy())
        # plot.colorbar()
        # plot.show()
        attn = nn.Softmax(-1)(score)
        attn.masked_fill_(attn_mask.bool(), 0)
        attn = torch.tensor(attn,requires_grad=True)
        # print(attn.requires_grad)
        # print('attn.shape:',attn.shape)
        # plot.matshow(attn[0,0,:,:].squeeze(0).squeeze(0).detach().numpy())
        # plot.colorbar()
        # plot.show()
        context = torch.matmul(attn, V)
        return context.to(device)


# 这里是graph_attention的核心操作：包含计算Q，K，V，同时为了只关注相邻结点所以添加了blockM 作为mask, 同时还分了多头
class MultiHeadAttention(nn.Module):
    # 这个算法中包含的multiheadattention 包含三个部分
    # graph-self attention --> 使用的mask 矩阵是邻接矩阵blockM
    # 输出序列的attention--> 使用的mask 是固定的就是还没有到的进行mask
    # 输入和输出的attention--> 应该是没有使用mask 的
    def __init__(self):
        super(MultiHeadAttention, self).__init__()  # 在此定义需要学习的参数，Q,K,V, 的参数，用的全是全连接层
        self.W_Q = nn.Linear(d_model, d_Q * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_Q * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_V * n_head, bias=False)
        self.fc = nn.Linear(d_V * n_head, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):  # 这里就能解释为什么需要三个参数了，
        # 虽然在graph attention中都是enc_embedding,但是其它功能就不一样了
        residual, batch_size = input_Q, input_Q.shape[0]  # residual  是为了实现层级之间的传递
        Q = self.W_Q(input_Q).view(batch_size, -1, n_head, d_Q).transpose(1, 2)  # [batch_size,n_head,n_of_node,d_Q]
        K = self.W_K(input_K).view(batch_size, -1, n_head, d_Q).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_head, d_V).transpose(1, 2)
        # Mask 也要进行进行维度的变化， 加在head 那个维度，复制n_head次
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context = ScaledDotProductAttention()(Q, K, V, attn_mask)  # todo   这里不再保存attn 了
        # 将多个头拼接起来  三维[batch_size, no_of_node, d_Q*n_head]
        context = context.transpose(1, 2).reshape(batch_size, -1, d_Q * n_head)
        # 全连接层转换回初始的维度
        out_put = self.fc(context)
        return nn.LayerNorm(d_model).to(device)(out_put + residual)


# 这个函数是在做什么呢？   为了实现参数的共享？
# Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_sef_atten_0 = MultiHeadAttention()   # 这个是linkM 的multi-head attention
        self.enc_self_attn_1 = MultiHeadAttention()   # 分为六个阻挡方向
        self.enc_self_attn_2 = MultiHeadAttention()   # 不同阻挡方向系数不同，贡献不同
        self.enc_self_attn_3 = MultiHeadAttention()
        self.enc_self_attn_4 = MultiHeadAttention()
        self.enc_self_attn_5 = MultiHeadAttention()
        self.enc_self_attn_6 = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_embedding, enc_graph_attn_mask, enc_link_attn_mask): # 这里的enc_graph-attn_mask 是true false
                                                           # 维度为batch_size* 6* n_of_node_n_of_node
        enc_link_embedding_out=self.enc_sef_atten_0(enc_embedding, enc_embedding, enc_embedding,
                                               enc_link_attn_mask)
        enc_graph_attn_mask_1=enc_graph_attn_mask[:,0,:,:]  # 保留batch_size,选取阻挡关系1，降到3维
        enc_embedding_out_1 = self.enc_self_attn_1(enc_embedding, enc_embedding, enc_embedding,
                                               enc_graph_attn_mask_1)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_graph_attn_mask_2=enc_graph_attn_mask[:,1,:,:]  
        enc_embedding_out_2 = self.enc_self_attn_2(enc_embedding, enc_embedding, enc_embedding,
                                               enc_graph_attn_mask_2)      
        enc_graph_attn_mask_3=enc_graph_attn_mask[:,2,:,:]  
        enc_embedding_out_3 = self.enc_self_attn_3(enc_embedding, enc_embedding, enc_embedding,
                                               enc_graph_attn_mask_3) 
        enc_graph_attn_mask_4=enc_graph_attn_mask[:,3,:,:]  
        enc_embedding_out_4 = self.enc_self_attn_4(enc_embedding, enc_embedding, enc_embedding,
                                               enc_graph_attn_mask_4) 
        enc_graph_attn_mask_5=enc_graph_attn_mask[:,4,:,:]  
        enc_embedding_out_5 = self.enc_self_attn_5(enc_embedding, enc_embedding, enc_embedding,
                                               enc_graph_attn_mask_5) 
        enc_graph_attn_mask_6=enc_graph_attn_mask[:,5,:,:]  
        enc_embedding_out_6 = self.enc_self_attn_6(enc_embedding, enc_embedding, enc_embedding,
                                               enc_graph_attn_mask_6) 
        enc_embedding_out=enc_embedding_out_1+enc_embedding_out_2+enc_embedding_out_3+enc_embedding_out_4 +\
            enc_embedding_out_5+enc_embedding_out_6 +enc_link_embedding_out

        enc_embedding_out = self.pos_ffn(enc_embedding_out)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_embedding_out.to(device)


def graph_attn_mask(blockM):
    mask = blockM.data.eq(0)
    return mask.to(device)


def get_attn_subsequence_mask(dec_embedding):
    """
    seq: [batch_size, n_of_node, d_model]
    """
    # 需要dec_embedding提供的信息只包含batch_size 和n_of_node
    attn_shape = [dec_embedding.size(0), dec_embedding.size(1), dec_embedding.size(1)]
    # attn_shape: [batch_size,n_of_node, n_of_node]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    # print('subsequene_mask:',subsequence_mask)
    return subsequence_mask.to(device)  # [batch_size, tgt_len, tgt_len]


def get_attn_pad_mask(position_1, position_2):  # 只用于cross attention 但是我们的数据等长那么就是一个全为false的方阵了
    """          # 在 cross attention 中先dec (seq ) 再 enc(graph)
    position_1: [batch_size, n_of_node,d_feature]
    position_2: [batch_size, n_of_node, d_feature]
    """
    batch_size, len_q = position_1.shape[0], position_1.shape[1]  # 这个seq_q只是用来expand维度的   graph
    batch_size, len_k = position_2.shape[0], position_2.shape[1]  # seq
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    a = position_1[:, :, 1].data.eq(-1).unsqueeze(1).transpose(-2, -1)  # [batch_size, n_of_node ]
    # print('a.shape:',a.shape)
    # print('a:',a)
    b = position_2[:, :, 1].data.eq(-1).unsqueeze(1)  # [batch_zise, n_of_node ]
    a = a.repeat(1, 1, len_k)  # 1*20*(3)
    b = b.repeat(1, len_q, 1)  # 1*20*(3)
    # #b = torch.full([batch_size,len_q, len_k], 1)   这个是没有加padding 的
    # pad_attn_mask = b.data.eq(0)  # [batch_size, 1, len_k], True is masked    等于0 的地方是padding,padding 都被mask掉了，true is mask
    # # print(pad_attn_mask)
    return (a | b).to(device)  # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos2emb = position2embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        # self. embedding layer!

    def forward(self, position,blockM,linkM):
        enc_pos_embeddings = self.pos2emb(position)
        enc_graph_attn_mask = graph_attn_mask(blockM)
        enc_link_attn_mask= graph_attn_mask(linkM)
        for layer in self.layers:
            enc_embeddings = layer(enc_pos_embeddings, enc_graph_attn_mask, enc_link_attn_mask)
        return enc_embeddings.to(device)


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_embedding, enc_embedding, dec_self_attn_mask, dec_enc_attn_mask):
        dec_embedding = self.dec_self_attn(dec_embedding, dec_embedding, dec_embedding, dec_self_attn_mask)
        dec_embedding = self.dec_enc_attn(dec_embedding, enc_embedding, enc_embedding, dec_enc_attn_mask)
        dec_embedding = self.pos_ffn(dec_embedding)
        return dec_embedding.to(device)


class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position_eco = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 上面的2 表示间隔为2  div_term  指数下降
        pe[:, 0::2] = torch.sin(position_eco * div_term)  # 从0开始，间隔为2
        pe[:, 1::2] = torch.cos(position_eco * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 表示这是一个持久态，不会有梯度传播给他，但是能保存在buffer中

    def forward(self, x):  # x在输入之前进行了转置  [batch_size, n_of_node, d_model]-->[n_of_node, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]  # 截取从0到n_of_node 这一段
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pos2emb = position2embedding()
        self.pos_enc = PositionEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, position_seq, position_graph, enc_embedding):
        dec_embedding = self.pos2emb(
            position_seq)  # 将坐标转化成embedding   dec_output   dec_input transformer——with embedding中一样
        # decoder 是要有顺序的！所以要加入position encoding
        dec_embedding = self.pos_enc(dec_embedding.transpose(0, 1)).transpose(0, 1)
        # 已经是一个序列了，下面得到sequence mask
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_embedding)  # decoder_self——attention
        # dec_self_atten_pad_mask_attention   padding mask
        dec_self_attn_pad_mask = get_attn_pad_mask(position_seq, position_seq)
        dec_self_attn_mask = dec_self_attn_subsequence_mask | dec_self_attn_pad_mask
        # plot.matshow(dec_self_attn_mask[0,:,:])
        # plot.colorbar()
        # plot.show()
        dec_enc_attn_mask = get_attn_pad_mask(position_seq, position_graph)  # todo 这里的bug, qio!
        # 然后就要近入层级结构了：
        for layer in self.layers:
            dec_embedding = layer(dec_embedding, enc_embedding, dec_self_attn_mask, dec_enc_attn_mask)

        return dec_embedding.to(device)


class myGraphTransformer(nn.Module):
    def __init__(self):
        super(myGraphTransformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.embedding2position = nn.Linear(d_model, d_feature)  # 将输出的embedding转化成坐标  就不单独写函数了，这是position2embedding的反过程

    def forward(self, position_graph, blockM, linkM, position_seq):  
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        enc_embedding = self.encoder(position_graph, blockM,linkM)   
        dec_embedding = self.decoder(position_seq, position_graph, enc_embedding)
        position_output = self.embedding2position(dec_embedding)
        return position_output.to(device)


def criterion(position_output, position_real):  # 平均平方欧式距离  越小越好 todo 可能还得改回来！
    loss = 0
    # print(position_output)
    # print(position_real)
    for i in range(position_output.shape[0]):
        padding_loc = torch.nonzero(position_real[i,:,0] == -1)
        len_of_dec_input = padding_loc[0, 0]
        # print('===================len_of_dec_input===============')
        # print(len_of_dec_input)
        diff = position_output[i, 0:len_of_dec_input, :] - position_real[i, 0:len_of_dec_input, :]
        # print(diff)
        # print(diff.pow(2).sum())
        loss = loss + diff.pow(2).sum() / position_output.shape[0] / len_of_dec_input  # 这里应该除以len_of_dec_input
    return loss


model = myGraphTransformer().to(device)
model = torch.nn.DataParallel(model)
BP = 0
# path='save_epoch/p2p_blockM_linkM_position_type/epoch'+str(BP)+'.pickle'
# model.load_state_dict(torch.load(path))
LR=1e-3
# criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.85)  # 用adam的话效果不好
linkM, blockM, position_graph, position_seq, position_real,node_type, order = make_data(n_iter)
loader = Data.DataLoader(Mydataset(linkM, blockM, position_graph, position_seq, position_real,node_type, order), batch_size, True)
start_time = time.time()
loss_save = []

# setup_seed(42)
# seed_torch(42)

for epoch in range(epochs):
    for linkM, blockM, position_graph, position_seq, position_real, node_type, order in loader:
        position = model(position_graph, blockM,linkM, position_seq)
        loss = criterion(position, position_real)
        print('Epoch:', '%04d' % (BP + epoch + 1), 'loss =', '{:.6f}'.format(loss))
        txt_file = open('loss/loss_p2p_blockM_linkM_position.txt', "a", encoding="utf-8")
        txt_file.write(str(loss.cpu().detach().numpy()))
        txt_file.write("\n")
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        #  保存中间模型，以免意外中断从而需要重新训练。
    if epoch % 100 == 1:
        for p in optimizer.param_groups:
            p['lr'] *= 0.95        
        path = 'save_epoch/p2p_blockM_linkM_position/epoch' + str(BP + epoch - 1) + '.pickle'
        torch.save(model.state_dict(), path)

end_time = time.time()
time_used = end_time - start_time
print('time cost', time_used, 's')
# ---------------------------------------------------------------------------
path = 'saved_model/p2p_blockM_linkM_position.pickle'
torch.save(model.state_dict(), path)

