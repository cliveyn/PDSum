import torch
import numpy as np
from scipy.sparse import vstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


class Model(torch.nn.Module):
    def __init__(self, D_in, D_hidden, head, dropout=0.0):
        super(Model, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim=D_in, num_heads=head, dropout=dropout, batch_first=True)
        self.layernorm = torch.nn.LayerNorm(D_in)
        self.embd = torch.nn.Linear(D_in,D_hidden)
        self.attention = torch.nn.Linear(D_hidden,1)
        
    def forward(self, x_org, mask=None):
        x, mha_w = self.mha(x_org,x_org,x_org,key_padding_mask=mask)
        x = self.layernorm(x_org+x)
        
        x = self.embd(x)
        x = torch.tanh(x) 
        a = self.attention(x)
        if mask is not None:
            a = a.masked_fill_((mask == 1).unsqueeze(-1), float('-inf'))
        w = torch.softmax(a, dim=1)
        o = torch.matmul(w.permute(0,2,1), x)
        return o, mha_w, w, x

def get_loss(sample_outputs, class_indices, class_embds, temp):
    k = torch.exp(torch.nn.functional.cosine_similarity(sample_outputs[:,None], class_embds, axis=2)/temp)
    loss = -1*(torch.log(k[np.arange(len(class_indices)),class_indices]/k.sum(1))).sum()
    return loss

def get_cluster_theme(all_vocab, window, N):
    cluster_ids = list(window['Query'].unique())
    
    cluster_tf_dic = {}
    for cluster_id in cluster_ids:
            cluster_tf_dic[cluster_id] = np.sum(window[window['Query']==cluster_id].article_TF)  
            
    cluster_tf = vstack(cluster_tf_dic.values())
    cluster_df = np.bincount(cluster_tf.indices, minlength=cluster_tf.shape[1]).reshape(1,-1)
    cluster_idf = np.log((len(cluster_ids)+1)/(cluster_df+1))+1 #scikit-learn formual = log((N+1)/(df+1))+1
    
    cluster_keyword_score_all = cluster_tf.multiply(cluster_idf).tocsr()
    
    cluster_topN_indices = {}
    cluster_topN_words = {}
    cluster_topN_scores = {}
    for i in range(len(cluster_ids)):
        cluster_id = cluster_ids[i]        
        topN_indices = cluster_keyword_score_all[i].indices[cluster_keyword_score_all[i].data.argsort()[:-(N+1):-1]]
        cluster_topN_indices[cluster_id] = topN_indices
         
        cluster_topN_words[cluster_id] = [all_vocab[k] for k in cluster_topN_indices[cluster_id]]
        cluster_topN_scores[cluster_id] = cluster_keyword_score_all[i][:,cluster_topN_indices[cluster_id]].toarray().squeeze()

    return cluster_topN_indices, cluster_topN_scores, cluster_topN_words

def masking(df, idx, num_sens = 50):
    org_embd = torch.tensor(df.loc[idx,'sentence_embds'][:num_sens])
    maksed_embd = torch.zeros(num_sens, org_embd.shape[1])
    mask = torch.ones(num_sens)
    maksed_embd[:org_embd.shape[0], :] = org_embd
    mask[:org_embd.shape[0]] = 0
    
    return maksed_embd, mask

def initialize(df_org):
    st_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1').cuda() 
    embeddings = []
    for sentences in df_org['sentences']:
        embedding = st_model.encode(sentences)
        embeddings.append(embedding)
    df_org['sentence_embds'] = embeddings
    
    masked = [masking(df_org, idx) for idx in df_org.index]
    masked_tensors = torch.stack([m[0] for m in masked]).cuda()
    masks = torch.stack([m[1] for m in masked]).cuda()
    mean_embds = torch.div(masked_tensors.sum(1),(1-masks).sum(1).reshape(-1,1)).cpu().detach().numpy()
    
    df_org['mean_embd'] = list(mean_embds)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), tokenizer=lambda x: x, lowercase=False, norm=None)
    tfidf_vectorizer.fit_transform([sum(k, []) for k in df_org['sentence_tokens']])
    all_vocab = tfidf_vectorizer.get_feature_names()

    count_vectorizer = CountVectorizer(tokenizer=lambda x: x, ngram_range = (1,2), vocabulary = list(all_vocab), lowercase=False)
    df_org['sentence_TFs'] = [count_vectorizer.transform(y) for y in df_org['sentence_tokens'].values]
    df_org['article_TF'] = [sum(a) for a in df_org['sentence_TFs'].values]

    return df_org, masked_tensors, masks, all_vocab