from evaluate import load
import spacy
import pandas as pd
import collections

bertscore = load("bertscore")
nlp = spacy.load("en_core_web_lg")

def get_rouge_score(target_tokens, prediction_tokens, n = 1):
  if n == 'L':
    return _score_lcs(target_tokens, prediction_tokens)
  else:
    target_ngrams = _create_ngrams(target_tokens, n)
    prediction_ngrams = _create_ngrams(prediction_tokens, n)
    return _score_ngrams(target_ngrams, prediction_ngrams)  


def get_daily_score_df(summary_df, true_summaries, measure, n = 1, W2E_concurrent_queries_df = None):
    if (type(n) != int) and (n!='L'):
        return "Error: 'n' must be integer or 'L'"

    output_score_df = pd.DataFrame(index=list(summary_df.index))
    
    for query in summary_df.index:
        true_summary = true_summaries[true_summaries.Query==query].tokenized_summary.values[0]
        for date in summary_df.loc[query].dropna().index:
            if len(W2E_concurrent_queries_df) > 0: #W2E
                true_summary = true_summaries[(true_summaries.Query==query) & (true_summaries.date == date)].tokenized_summary.values[0]
            
            predicted_summary = summary_df.loc[query,date]

            if measure == 'ROUGE':
                output_score_df.loc[[query], date] = pd.Series([list(get_rouge_score(true_summary, predicted_summary, n))], index=[query])
            elif measure == 'BERT':
                if len(predicted_summary) > 0:
                    output_score_df.loc[[query], date] = pd.Series([sum(list(bertscore.compute(predictions=[' '.join(predicted_summary)], references=[' '.join(true_summary)], lang="en").values())[:3], [])], index=[query])
                else:
                    output_score_df.loc[[query], date] = pd.Series([[0, 0, 0]], index=[query])
    return output_score_df

def get_daily_contrast_df(summary_df, true_summaries, measure, n = 1, W2E_concurrent_queries_df = None):
    if (type(n) != int) and (n!='L'):
        return "Error: 'n' must be integer or 'L'"

    output_score_df = pd.DataFrame(index=list(summary_df.index))
    
    for date in true_summaries.date.unique():
        date = str(date)[:10]
        if date not in summary_df.columns: continue
        if len(W2E_concurrent_queries_df) > 0: #W2E
            target_true_summaries = {}
            for query in W2E_concurrent_queries_df.loc[date].values[0]:
                if query not in summary_df.index: continue
                for target_date in summary_df.loc[query].dropna().index:
                    if target_date >= date:
                        target_true_summaries[query] = summary_df.loc[query, target_date]
                        break                
        else:
            target_true_summaries = true_summaries[true_summaries.date==date].summary.to_dict()

        for query in summary_df[date].dropna().index:
            summary = summary_df.loc[query,date]
            scores = []
            for target_query in target_true_summaries:
                if query == target_query:
                    continue
                true_summary = target_true_summaries[target_query]
                if measure == 'ROUGE':
                    score = list(get_rouge_score(true_summary, summary, n))
                else:
                    score = sum(list(bertscore.compute(predictions=[' '.join(summary)], references=[' '.join(true_summary)], lang="en").values())[:3], [])
                scores.append(score)
            if len(scores) > 0:
                output_score_df.loc[[query], date] = pd.Series([scores], index=[query])
    return output_score_df    


def get_novel_overlap_score_df(summary_df, true_summaries, measure, n = 1):
    if (type(n) != int) and (n!='L'):
        return "Error: 'n' must be integer or 'L'"

    output_novel_df = pd.DataFrame(index=list(summary_df.index))
    output_overlap_df = pd.DataFrame(index=list(summary_df.index))
    novel_ratio_df = pd.DataFrame(index=list(summary_df.index))
    
    for query in summary_df.index:
        true_summary = sum(true_summaries[true_summaries.Query==query].tokenized_summary.values, [])
        # if 'topic' in true_summaries: #W2E
        #     true_summary = true_summaries[(true_summaries.Query==query) & (true_summaries.date == prev_date)].tokenized_summary.values[0]

        existing_dates = summary_df.loc[query].dropna().index
        
        #prev_summary = []
        for i in range(len(existing_dates)-1) :
            prev_date = existing_dates[i]
            prev_summary = summary_df.loc[query, prev_date]
            #prev_summary = prev_summary + summary_df.loc[query, prev_date]
            
            for j in range(i+1,len(existing_dates)):
                new_date = existing_dates[j]
                new_summary = summary_df.loc[query, new_date]
                # if 'topic' in true_summaries: #W2E
                #     true_summary = true_summaries[(true_summaries.Query==query) & (true_summaries.date == new_date)].tokenized_summary.values[0]

                novel_score, overlap_score, novel_ratio = get_novel_overlap_score(true_summary, prev_summary, new_summary, measure, n)
                output_novel_df.loc[[query], new_date] = pd.Series([list(novel_score)], index=[query])
                output_overlap_df.loc[[query], new_date] = pd.Series([list(overlap_score)], index=[query])
                novel_ratio_df.loc[[query], new_date] = pd.Series([novel_ratio], index=[query])
    
    return output_novel_df, output_overlap_df, novel_ratio_df    


def get_novel_overlap_score(target_tokens, prev_prediction_tokens, new_prediction_tokens, measure, n = 1):
  if (type(n) != int) and (n!='L'):
        return "Error: 'n' must be integer or 'L'"
  
  overlap_prediction_tokens = []
  novel_prediction_tokens = new_prediction_tokens.copy()
  for token1 in prev_prediction_tokens:
    if token1 in novel_prediction_tokens:
      novel_prediction_tokens.remove(token1)
      overlap_prediction_tokens.append(token1)

  if (len(novel_prediction_tokens)+len(overlap_prediction_tokens)) > 0:
    novel_ratio = len(novel_prediction_tokens)/(len(novel_prediction_tokens)+len(overlap_prediction_tokens))
  else:
    novel_ratio = 0
    
  if measure == 'ROUGE':
    novel_rouge_score = get_rouge_score(target_tokens, novel_prediction_tokens, n)
    overlap_rouge_score = get_rouge_score(target_tokens, overlap_prediction_tokens, n)
    return novel_rouge_score, overlap_rouge_score, novel_ratio
  elif measure == 'BERT':
    if len(novel_prediction_tokens) > 0:
      novel_bert_score = sum(list(bertscore.compute(predictions=[' '.join(novel_prediction_tokens)], references=[' '.join(target_tokens)], lang="en").values())[:3], [])
    else:
      novel_bert_score = [0, 0, 0]
    if len(overlap_prediction_tokens) > 0:
      overlap_bert_score = sum(list(bertscore.compute(predictions=[' '.join(overlap_prediction_tokens)], references=[' '.join(target_tokens)], lang="en").values())[:3], [])
    else:
      overlap_bert_score = [0, 0, 0]
    return novel_bert_score, overlap_bert_score, novel_ratio    

def get_tokens(sentence):
    parsed = nlp(sentence)
    tokens = []
    for s in parsed.sents:
        tokens = tokens + [token.lemma_.lower() for token in s if (token.text.isalnum() and not token.is_stop and not token.is_punct)]  #and not token.like_num
    return tokens
def get_tokenized_summary(summary_df):
    tokenized_summary = pd.DataFrame(columns = summary_df.columns)
    for (idx, row) in summary_df.iterrows():
        for date in row.dropna().index:
            tokenized_summary.loc[idx, date] = get_tokens(row[date])
    return tokenized_summary
    

#For the below four methods, the base code is from https://github.com/google-research/google-research/tree/master/rouge
#######################################################################################################################
def _lcs_table(ref, can):
  rows = len(ref)
  cols = len(can)
  lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
  for i in range(1, rows + 1):
    for j in range(1, cols + 1):
      if ref[i - 1] == can[j - 1]:
        lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
      else:
        lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
  return lcs_table

def _score_lcs(target_tokens, prediction_tokens):
  lcs_table = _lcs_table(target_tokens, prediction_tokens)
  lcs_length = lcs_table[-1][-1]

  if len(prediction_tokens) == 0 or len(target_tokens) == 0:
    precision = 0
    recall = 0
  else:
    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(target_tokens)
    
  if precision + recall > 0:
    fmeasure = 2 * precision * recall /(precision + recall)
  else:
    fmeasure = 0

  return (precision, recall, fmeasure)  

def _score_ngrams(target_ngrams, prediction_ngrams):
  intersection_ngrams_count = 0
  for ngram in target_ngrams:
    intersection_ngrams_count += min(target_ngrams[ngram],
                                     prediction_ngrams[ngram])
  target_ngrams_count = sum(target_ngrams.values())
  prediction_ngrams_count = sum(prediction_ngrams.values())

  precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
  recall = intersection_ngrams_count / max(target_ngrams_count, 1)
  if precision + recall > 0:
    fmeasure = 2 * precision * recall /(precision + recall)
  else:
    fmeasure = 0

  return (precision, recall, fmeasure)  

def _create_ngrams(tokens, n):
  ngrams = collections.Counter()
  for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
    ngrams[ngram] += 1
  return ngrams  
#######################################################################################################################    