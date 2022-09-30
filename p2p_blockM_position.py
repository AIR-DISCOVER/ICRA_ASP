# _*_coding:utf-8 _*_
# @Time : 2022/9/4 
# @Author : Lin

# node feature: position
# edge feature: block 

import pickle
import torch
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import math
import torch.optim as optim
import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



n_iter = 45 
n_doc=60
batch_size = n_iter  
d_Q = 64  
d_V = 64 
num_of_node = 45 
d_feature = 3  
d_node_type=24 
d_model = 256  
n_layers = 6  
n_head = 1  
d_ff = 256  
epochs = 10000 


def position_normal(position):
    position[:,0]=position[:,0]/max(position[:,0])
    position[:, 1] = position[:,1] / max(position[:, 1])
    position[:, 2] = position[:,2] / max(position[:, 2])
    return position

def read_data(filename):
    f = open(filename, 'rb')
    data=pickle.load(f)
    linkM=data['linkM']
    position=data['position']
    position=position_normal(position)
    node_type=data['node_type']   
    order=data['order']
    blockM=data['blockM']  
    blockM=torch.transpose(blockM, 0,2)
    node_type=node_type[order.long()]
    output_pos=position[order.long()]  
    return linkM, blockM,position,node_type, output_pos, order

def shuffle_read_train(filename): #filename='shuffle.txt'
    f=open(filename, 'r')
    a=f.read().split(',')
    sh_list=[]
    for i in a:
        sh_list.append(int(i))
    return sh_list 

def make_data(n_iter): 
    filename_prefix = 'data/LEGO'
    filename_suffix = '.data'
    linkM = torch.empty(0, num_of_node, num_of_node)  
    position_enc_input = torch.empty(0, num_of_node, d_feature)
    position_dec_input = torch.empty(0, num_of_node, d_feature)
    position_dec_output = torch.empty(0, num_of_node, d_feature)
    node_type_enc_input=torch.empty(0,num_of_node,d_node_type)
    blockM =torch.empty(0, 6, num_of_node,num_of_node)  
    order=torch.empty(0,num_of_node).long()
    stl_list=shuffle_read_train('shuffle.txt') 
    for i in range(n_iter):
        filename = filename_prefix + str(stl_list[i]-1) + filename_suffix
        linkM_item,blockM_item, position_item, node_type_item, output_pos_item, order_item = read_data(filename)
        padding_len = num_of_node - linkM_item.shape[0]
        order_added = np.ones([1,num_of_node])*-1
        order_added[0,0:len(order_item)] = order_item
        order_added=torch.tensor(order_added)
        order_added = order_added
        order=torch.cat([order,order_added.long()],dim=0)    
        linkM_right = torch.zeros([linkM_item.shape[0], padding_len])
        linkM_item = torch.cat([linkM_item, linkM_right], dim=1)
        linkM_bottom = torch.zeros([padding_len, linkM_item.shape[1]])
        linkM_item = torch.cat([linkM_item, linkM_bottom], dim=0)
        linkM = torch.cat([linkM, linkM_item.unsqueeze(0)], dim=0)   
        blockM_right=torch.zeros([6,blockM_item.shape[1],padding_len])
        blockM_item=torch.cat([blockM_item, blockM_right],axis=2)
        blockM_bottom=torch.zeros([6, padding_len, blockM_item.shape[2]])
        blockM_item=torch.cat([blockM_item,blockM_bottom],axis=1)
        blockM=torch.cat([blockM,blockM_item.unsqueeze(0)],dim=0)  
        position_added = torch.cat([position_item, torch.ones(padding_len, d_feature) * -1], dim=0)  
        position_enc_input = torch.cat([position_enc_input, position_added.float().unsqueeze(0)], dim=0)  
        node_type_add=torch.cat([node_type_item,torch.ones(padding_len, d_node_type)*-1],dim=0) 
        node_type_enc_input=torch.cat([node_type_enc_input, node_type_add.float().unsqueeze(0)],dim=0) 
        position_added = torch.cat([torch.ones(1, d_feature) * -1, output_pos_item], dim=0)
        position_added = torch.cat([position_added, torch.ones(padding_len - 1, d_feature) * -1], dim=0)
        position_dec_input = torch.cat([position_dec_input, position_added.float().unsqueeze(0)], dim=0)  
        position_added = torch.cat([output_pos_item, torch.ones(padding_len, d_feature) * -1], dim=0)  
        position_dec_output = torch.cat([position_dec_output, position_added.float().unsqueeze(0)], dim=0) 

    return linkM, blockM, position_enc_input, position_dec_input, position_dec_output,node_type_enc_input, order


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
        score.masked_fill_(attn_mask.bool(), -1e9)  
        attn = nn.Softmax(-1)(score)
        attn.masked_fill_(attn_mask.bool(), 0)
        attn = torch.tensor(attn,requires_grad=True)
        context = torch.matmul(attn, V)
        return context.to(device)


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super(MultiHeadAttention, self).__init__() 
        self.W_Q = nn.Linear(d_model, d_Q * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_Q * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_V * n_head, bias=False)
        self.fc = nn.Linear(d_V * n_head, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):  
        residual, batch_size = input_Q, input_Q.shape[0] 
        Q = self.W_Q(input_Q).view(batch_size, -1, n_head, d_Q).transpose(1, 2) 
        K = self.W_K(input_K).view(batch_size, -1, n_head, d_Q).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_head, d_V).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context = ScaledDotProductAttention()(Q, K, V, attn_mask)  
        context = context.transpose(1, 2).reshape(batch_size, -1, d_Q * n_head)
        out_put = self.fc(context)
        return nn.LayerNorm(d_model).to(device)(out_put + residual)


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
        return nn.LayerNorm(d_model).to(device)(output + residual)  


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn_1 = MultiHeadAttention() 
        self.enc_self_attn_2 = MultiHeadAttention()  
        self.enc_self_attn_3 = MultiHeadAttention()
        self.enc_self_attn_4 = MultiHeadAttention()
        self.enc_self_attn_5 = MultiHeadAttention()
        self.enc_self_attn_6 = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_embedding, enc_graph_attn_mask):

        enc_graph_attn_mask_1=enc_graph_attn_mask[:,0,:,:] 
        enc_embedding_out_1 = self.enc_self_attn_1(enc_embedding, enc_embedding, enc_embedding,
                                               enc_graph_attn_mask_1) 
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
            enc_embedding_out_5+enc_embedding_out_6 

        enc_embedding_out = self.pos_ffn(enc_embedding_out)
        return enc_embedding_out.to(device)


def graph_attn_mask(blockM):
    mask = blockM.data.eq(0)
    return mask.to(device)


def get_attn_subsequence_mask(dec_embedding):

    attn_shape = [dec_embedding.size(0), dec_embedding.size(1), dec_embedding.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) 
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask.to(device)  


def get_attn_pad_mask(position_1, position_2):  
    len_q =position_1.shape[1]  
    len_k =position_2.shape[1] 
    a = position_1[:, :, 1].data.eq(-1).unsqueeze(1).transpose(-2, -1) 
    b = position_2[:, :, 1].data.eq(-1).unsqueeze(1)  
    a = a.repeat(1, 1, len_k)  
    b = b.repeat(1, len_q, 1)  
    return (a | b).to(device) 


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos2emb = position2embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, position,blockM):
        enc_pos_embeddings = self.pos2emb(position)
        enc_graph_attn_mask = graph_attn_mask(blockM)
        for layer in self.layers:
            enc_embeddings = layer(enc_pos_embeddings, enc_graph_attn_mask)
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
        pe[:, 0::2] = torch.sin(position_eco * div_term)
        pe[:, 1::2] = torch.cos(position_eco * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe) 

    def forward(self, x): 
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pos2emb = position2embedding()
        self.pos_enc = PositionEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, position_seq, position_graph, enc_embedding):
        dec_embedding = self.pos2emb(
            position_seq) 
        dec_embedding = self.pos_enc(dec_embedding.transpose(0, 1)).transpose(0, 1)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_embedding) 
        dec_self_attn_pad_mask = get_attn_pad_mask(position_seq, position_seq)
        dec_self_attn_mask = dec_self_attn_subsequence_mask | dec_self_attn_pad_mask
        dec_enc_attn_mask = get_attn_pad_mask(position_seq, position_graph) 
        for layer in self.layers:
            dec_embedding = layer(dec_embedding, enc_embedding, dec_self_attn_mask, dec_enc_attn_mask)

        return dec_embedding.to(device)


class myGraphTransformer(nn.Module):
    def __init__(self):
        super(myGraphTransformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.embedding2position = nn.Linear(d_model, d_feature)  

    def forward(self, position_graph, blockM, position_seq):  
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        enc_embedding = self.encoder(position_graph, blockM)   
        dec_embedding = self.decoder(position_seq, position_graph, enc_embedding)
        position_output = self.embedding2position(dec_embedding)
        return position_output.to(device)


def criterion(position_output, position_real): 
    loss = 0
    for i in range(position_output.shape[0]):
        padding_loc = torch.nonzero(position_real[i,:,0] == -1)
        len_of_dec_input = padding_loc[0, 0]
        diff = position_output[i, 0:len_of_dec_input, :] - position_real[i, 0:len_of_dec_input, :]
        loss = loss + diff.pow(2).sum() / position_output.shape[0] / len_of_dec_input 
    return loss


model = myGraphTransformer().to(device)
model = torch.nn.DataParallel(model)
BP = 0
# path='save_epoch/p2p_blockM_linkM_position_type/epoch'+str(BP)+'.pickle'
# model.load_state_dict(torch.load(path))
LR=1e-4
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.85) 
linkM, blockM, position_graph, position_seq, position_real,node_type, order = make_data(n_iter)
loader = Data.DataLoader(Mydataset(linkM, blockM, position_graph, position_seq, position_real,node_type, order), batch_size, True)
start_time = time.time()
loss_save = []

for epoch in range(epochs):
    for linkM, blockM, position_graph, position_seq, position_real, node_type, order in loader:
        position = model(position_graph, blockM, position_seq)
        loss = criterion(position, position_real)
        print('Epoch:', '%04d' % (BP + epoch + 1), 'loss =', '{:.6f}'.format(loss))
        txt_file = open('loss/loss_p2p_blockM_position.txt', "a", encoding="utf-8")
        txt_file.write(str(loss.cpu().detach().numpy()))
        txt_file.write("\n")
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 1:
        for p in optimizer.param_groups:
            p['lr'] *= 0.95
        path = 'save_epoch/p2p_blockM_position/epoch' + str(BP + epoch - 1) + '.pickle'
        torch.save(model.state_dict(), path)

end_time = time.time()
time_used = end_time - start_time
print('time cost', time_used, 's')
# ---------------------------------------------------------------------------
path = 'saved_model/p2p_blockM_position.pickle'
torch.save(model.state_dict(), path)
