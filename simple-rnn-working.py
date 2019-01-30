#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np


# In[2]:


data=open('./dataset/dino.txt','r').read()
data=data.lower()
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)
char_to_ix={char:ix for ix,char in enumerate(sorted(chars))}
ix_to_char={ix:char for ix,char in enumerate(sorted(chars))}
print(ix_to_char)


# In[38]:


H=200
seq_length=30
lr=1e-1
Whx=np.random.randn(H,vocab_size)*0.01
Whh=np.random.randn(H,H)*0.01
Why=np.random.randn(vocab_size,H)*0.01
bh=np.zeros((H,1))
by=np.zeros((vocab_size,1))


# In[39]:


def loss_fun(inputs,targets,hprev):
    xs,hs,ys,ps={},{},{},{}
    hs[-1]=hprev
    loss=0
    for t in range(len(inputs)):
        xs[t]=np.zeros((vocab_size,1))
        xs[t][inputs[t]]=1
        hs[t]=np.tanh(np.dot(Whx,xs[t])+np.dot(Whh,hs[t-1])+bh)
        ys[t]=np.dot(Why,hs[t])+by
        ps[t]=np.exp(ys[t])/np.sum(np.exp(ys[t]))
        loss+= -np.log(ps[t][targets[t],0])
        
    dWhx, dWhh, dWhy = np.zeros_like(Whx), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext 
        dhraw = (1 - hs[t] * hs[t]) * dh 
        dbh += dhraw
        dWhx += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWhx, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) 
    return loss, dWhx, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]


# In[40]:


def sample(hprev,seed,sl):
    six=[]
    x=np.zeros((vocab_size,1))
    x[seed]=1
    for i in range(sl):
        hprev=np.tanh(np.dot(Whx,x)+np.dot(Whh,hprev)+bh)
        y=np.dot(Why,hprev)+by
        p=np.exp(y)/np.sum(np.exp(y))
        sampled_number=np.random.choice(np.arange(vocab_size),p=p.ravel())
        six.append(sampled_number)
        x=np.zeros((vocab_size,1))
        x[sampled_number]=1
    return six


# In[41]:


n=0 #iteration number
p=0 #from what position to take the next set of characters
mWhx,mWhh,mWhy,mbh,mby=np.zeros_like(Whx),np.zeros_like(Whh),np.zeros_like(Why),np.zeros_like(bh),np.zeros_like(by)
smooth_loss=-np.log(1/vocab_size)*seq_length
while(True):
    if p+seq_length+1>=data_size or n==0:
        hprev=np.zeros((H,1))
        p=0
    inputs=[char_to_ix[char] for char in data[p:p+seq_length]]
    targets=[char_to_ix[char] for char in data[p+1:p+1+seq_length]]
    
    if n%100==0:
        sample_ix=sample(hprev,inputs[0],200)
        sample_text=''.join(ix_to_char[ix] for ix in sample_ix)
        print('---\n %s \n---'%sample_text)
        
    loss,dWhx,dWhh,dWhy,dbh,dby,hprev=loss_fun(inputs,targets,hprev)
    smooth_loss=smooth_loss*0.999 + loss*0.001
    if n%100==0:
        print('loss:%d after %d iterations'%(smooth_loss,n))
    for param,dparam,mem in zip([Whx,Whh,Why,bh,by],[dWhx,dWhh,dWhy,dbh,dby],[mWhx,mWhh,mWhy,mbh,mby]):
        mem+=dparam*dparam
        param+= -lr*dparam/np.sqrt(mem+1e-8)
    p+=seq_length
    n+=1


# In[ ]:




