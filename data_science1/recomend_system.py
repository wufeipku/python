from collections import defaultdict
import csv
import numpy
import random
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
import tensorrec
import logging
logging.getLogger().setLevel(logging.INFO)

# Open and read in the ratings file
print('Loading ratings')
with open('D:/data_science/ml-20m/ratings.csv', 'r') as ratings_file:
    ratings_file_reader = csv.reader(ratings_file)
    raw_ratings = list(ratings_file_reader)
    raw_ratings_header = raw_ratings.pop(0)
# Iterate through the input to map MovieLens IDs to new internal IDs
# The new internal IDs will be created by the defaultdict on insertion
movielens_to_internal_user_ids = defaultdict(lambda: len(movielens_to_internal_user_ids))
movielens_to_internal_item_ids = defaultdict(lambda: len(movielens_to_internal_item_ids))
for row in raw_ratings[0:100000]:
    row[0] = movielens_to_internal_user_ids[int(row[0])]
    row[1] = movielens_to_internal_item_ids[int(row[1])]
    row[2] = float(row[2])
n_users = len(movielens_to_internal_user_ids)
n_items = len(movielens_to_internal_item_ids)

# Shuffle the ratings and split them in to train/test sets 80%/20%
random.shuffle(raw_ratings) # Shuffles the list in-place  打乱顺序
cutoff = int(.8 * len(raw_ratings))
train_ratings = raw_ratings[:cutoff]
test_ratings = raw_ratings[cutoff:]

# This method converts a list of (user, item, rating, time) to a sparse matrix
def interactions_list_to_sparse_matrix(interactions):
    users_column, items_column, ratings_column, _ = zip(*interactions)
    return sparse.coo_matrix((ratings_column, (users_column, items_column)),
        shape=(n_users, n_items))
# Create sparse matrices of interaction data
sparse_train_ratings = interactions_list_to_sparse_matrix(train_ratings)
sparse_test_ratings = interactions_list_to_sparse_matrix(test_ratings)

# Construct indicator features for users and items
user_indicator_features = sparse.identity(n_users)
item_indicator_features = sparse.identity(n_items)
# Build a matrix factorization collaborative filter model
cf_model = tensorrec.TensorRec(n_components=5)
# Fit the collaborative filter model
print("Training collaborative filter")
cf_model.fit(interactions=sparse_train_ratings,
 user_features=user_indicator_features,
 item_features=item_indicator_features)

# Create sets of train/test interactions that are only ratings >= 4.0
sparse_train_ratings_4plus = sparse_train_ratings.multiply(sparse_train_ratings >= 4.0)
sparse_test_ratings_4plus = sparse_test_ratings.multiply(sparse_test_ratings >= 4.0)
# This method consumes item ranks for each user and prints out recall@10 train/test metrics
def check_results(ranks):
    train_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=sparse_train_ratings_4plus,
        predicted_ranks=ranks,
        k=10
    ).mean()
    test_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=sparse_test_ratings_4plus,
        predicted_ranks=ranks,
        k=10
    ).mean()
    print("Recall at 10: Train: {:.4f} Test: {:.4f}".format(train_recall_at_10,
    test_recall_at_10))
# Check the results of the MF CF model
print("Matrix factorization collaborative filter:")
predicted_ranks = cf_model.predict_rank(user_features=user_indicator_features,
 item_features=item_indicator_features)
check_results(predicted_ranks)

# Let's try a new loss function: WMRB
print("Training collaborative filter with WMRB loss")
ranking_cf_model = tensorrec.TensorRec(n_components=5,
 loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
ranking_cf_model.fit(interactions=sparse_train_ratings_4plus,
 user_features=user_indicator_features,
 item_features=item_indicator_features,
 n_sampled_items=int(n_items * .01))
# Check the results of the WMRB MF CF model
print("WMRB matrix factorization collaborative filter:")
predicted_ranks = ranking_cf_model.predict_rank(user_features=user_indicator_features,
 item_features=item_indicator_features)
check_results(predicted_ranks)