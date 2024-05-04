import numpy as np
import torch

def hit(ng_item, recommends):
    # Check if the ground truth item (ng_item) is in the recommended list (recommends)
    return 1 if ng_item in recommends else 0

def ndcg(ng_item, recommends):
    # Calculate the Discounted Cumulative Gain (DCG)
    dcg = 0
    for i, item in enumerate(recommends):
        if item == ng_item:
            dcg += 1 / np.log2(i + 2)  # i+2 to start indexing from 1
    # Calculate the Ideal Discounted Cumulative Gain (IDCG) for the given list length
    idcg = 1 / np.log2(2)  # Ideal DCG for a single relevant item at the top
    # Calculate NDCG as the ratio of DCG to IDCG
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg


def metrics(model, test_loader, top_k, device):
	HR, NDCG = [], []

	# iterate over the test_loader to evaluate the model on the test data
	for user, item, _ in test_loader:
		user = user.to(device)
		item = item.to(device)

		with torch.no_grad():
			# Obtain predictions from the model
			predictions = model(user, item)
		
		# get the indices of the top-k highest predicted scores
		_, indices = torch.topk(predictions, top_k)

		# extract the top-k recommended items based on the indices
		recommends = torch.take(item, indices).cpu().numpy().tolist()

		# Get the ground truth item (ng_item) for the user (leave-one-out evaluation)
		ng_item = item[0].item()  # Assumes only one item per user for evaluation
		HR.append(hit(ng_item, recommends))
		NDCG.append(ndcg(ng_item, recommends))

	# Compute the mean HR and mean NDCG across all users in the test dataset
	mean_hr = np.mean(HR)
	mean_ndcg = np.mean(NDCG)

    # Return the mean HR and mean NDCG as evaluation metrics
	return mean_hr, mean_ndcg