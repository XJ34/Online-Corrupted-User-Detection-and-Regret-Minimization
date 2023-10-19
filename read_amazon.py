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

    rf = open(file_name, encoding='utf-8')

    l = rf.readline()
    count = 0
    for line in rf:
        dicts = json.loads(line)
        row_index = 0
        col_index = 0
        user_id = dicts["reviewerID"]
        business_id = dicts["asin"]
        rating = dicts["overall"]

        if not user_id in user_dict:  # .has_key(user_id):
            user_dict[user_id] = len(user_dict)
        row_index = user_dict[user_id]

        if not business_id in business_dict:  # .has_key(business_id):
            business_dict[business_id] = len(business_dict)
        col_index = business_dict[business_id]

        # data_list.append(float(rating))
        if float(rating) > 3.0:
            data_list.append(1)
        else:
            data_list.append(0)

        row_indics_list.append(row_index)
        col_indics_list.append(col_index)

    data = np.asarray(data_list)
    rows = np.asarray(row_indics_list)
    cols = np.asarray(col_indics_list)
    s_m = spp.csr_matrix((data, (rows, cols)))
    s_m[s_m > 0] = 1
    print(s_m)
    return s_m


def get_reduced_concrete_matrix(user_num, business_num):
    s_m = load_sparse_matrix("Musical_Instruments_5.json")
    row_reduced_matrix, topk = extract_rows(user_num * 3, s_m)
    reduced_matrix = extract_cols(business_num, row_reduced_matrix)
    reduced_matrix, topk = extract_rows(user_num, reduced_matrix)
    return reduced_matrix.toarray(), topk


s_m = load_sparse_matrix('Musical_Instruments_5.json').toarray()
print(s_m.shape)
# # m, topk = get_reduced_concrete_matrix(1400, 800)
# #
# # print(m.shape)
# # print(m)
# # np.save('amazon1400user800item.npy', m)
# # np.save('amazontopk.npy', topk)
# top_k = np.load('amazontopk.npy').tolist()
# bot_index = []
#
#
# rf = open('Musical_Instruments_5.json', encoding='utf-8')
# user_dict = {}
# label_dict = {}
# for line in rf:
#     dicts = json.loads(line)
#     row_index = 0
#     col_index = 0
#     user_id = dicts["reviewerID"]
#     business_id = dicts["asin"]
#     rating = dicts["overall"]
#     helpful = dicts['helpful']
#
#     if not user_id in user_dict:  # .has_key(user_id):
#         user_dict[user_id] = len(user_dict)
#         if helpful[1] > 0:
#             label_dict[len(label_dict)] = helpful[0] / helpful[1]
#         else:
#             label_dict[len(label_dict)] = 2
# print(label_dict)
#
#
# for i in top_k:
#     if label_dict[i] <= 0.2:
#         bot_index.append(top_k.index(i))
# print(len(bot_index))
# print(bot_index)
# np.save('amazonbot.npy', np.array(bot_index))
# #np.save('yelpzipbot.npy', np.array(bot_index))
#
# #np.save("yelp_5000user_1000item.npy", m)
# # s_m = load_sparse_matrix("Musical_Instruments_5.json").toarray()
# # print(s_m.shape)