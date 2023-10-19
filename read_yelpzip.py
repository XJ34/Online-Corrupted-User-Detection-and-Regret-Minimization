import scipy.io as sio
import scipy.sparse as spp
import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
import json
import pandas as pd


def extract_rows(top_k, sparse_matrix):
    business_review_count = sparse_matrix.getnnz(axis=1)
    business_count = business_review_count.shape[0]
    top_k_index = np.argsort(business_review_count)[business_count - 1: business_count - 1 - top_k: -1]
    # top_k_index = np.random.choice(business_count, top_k, replace=False)
    matrix = spp.vstack([sparse_matrix.getrow(i) for i in top_k_index])
    return matrix, top_k_index


def extract_cols(top_k, sparse_matrix):
    user_review_count = sparse_matrix.getnnz(axis=0)
    user_count = user_review_count.shape[0]

    top_k_index = np.argsort(user_review_count)[user_count - 1: user_count - 1 - top_k:-1]
    # top_k_index=np.random.choice(user_count, top_k, replace=False)
    matrix = spp.hstack([sparse_matrix.getcol(i) for i in top_k_index])
    return matrix


def load_sparse_matrix(file_name):
    data_list = []
    row_indics_list = []
    col_indics_list = []

    user_dict = {}
    business_dict = {}

    yelp = pd.read_excel(file_name)
    print(yelp.index)
    for row in yelp.index.values:
        user_id = yelp.iloc[row, 0]
        business_id = yelp.iloc[row, 1]
        rating = yelp.iloc[row, 2]
        if rating > 3:
            rating = 1
        else:
            rating = 0
        # label = yelp.iloc[row, 3]

        if not user_id in user_dict:  # .has_key(user_id):
            user_dict[user_id] = len(user_dict)
        row_index = user_dict[user_id]

        if not business_id in business_dict:  # .has_key(business_id):
            business_dict[business_id] = len(business_dict)
        col_index = business_dict[business_id]

        data_list.append(float(rating))

        row_indics_list.append(row_index)
        col_indics_list.append(col_index)

    data = np.asarray(data_list)
    rows = np.asarray(row_indics_list)
    cols = np.asarray(col_indics_list)
    s_m = spp.csr_matrix((data, (rows, cols)))
    return s_m


def get_reduced_concrete_matrix(user_num, business_num):
    s_m = load_sparse_matrix("YelpZip/yelpzip.xlsx")
    row_reduced_matrix, top_k_index = extract_rows(user_num * 3, s_m)
    reduced_matrix = extract_cols(business_num, row_reduced_matrix)
    reduced_matrix, top_k_index = extract_rows(user_num, reduced_matrix)
    return reduced_matrix.toarray(), top_k_index


#
# m, top_k_index = get_reduced_concrete_matrix(5000, 5000)
# print(m.shape)
# print(m)
# np.save('yelpzip5000user5000item.npy', m)
# np.save('yelpziptopk.npy', top_k_index)
# print(top_k_index)
# print(m.shape)
# print(m)
#
top_k = np.load('yelpziptopk.npy').tolist()
bot_index = []

yelp = pd.read_excel('YelpZip/yelpzip.xlsx')
user_dict = {}
label_dict = {}
for row in yelp.index.values:
    user_id = yelp.iloc[row, 0]
    label = yelp.iloc[row, 3]

    if not user_id in user_dict:  # .has_key(user_id):
        user_dict[user_id] = len(user_dict)
        label_dict[len(label_dict)] = label

for i in top_k:
    if label_dict[i] == -1:
        bot_index.append(top_k.index(i))
print(len(bot_index))
print(bot_index)
np.save('yelpzipbot.npy', np.array(bot_index))
