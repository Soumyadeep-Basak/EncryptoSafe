from utils import fetchEtherscanAPI
from fastapi import HTTPException
from models.ethereum import ModelParams
import pickle
import os
import numpy as np
from cachetools import cached, TTLCache

model_path = 'services/ml_score/xgb_5_model.pickle'

# Load the model
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    raise HTTPException(status_code=500, detail="Model file not found")


# Helper functions to get data from API
async def get_wallet_balance(eth_address: str) -> float:
    response = await fetchEtherscanAPI.fetch_eth_wallet_balance(eth_address)
    if response.get('status') == '1':
        balance_in_wei = response.get("result")
        return float(balance_in_wei) / 1e18
    else:
        raise HTTPException(status_code=response.status_code, detail="Error fetching wallet balance")


async def get_transactions(eth_address: str) -> list:
    response = await fetchEtherscanAPI.fetch_eth_wallet_transactions(eth_address)
    if response.get('status') == '1':
        return response.get("result", [])
    else:
        raise HTTPException(status_code=response.status_code, detail="Error fetching transactions")


# Fetch and engineer features
async def fetch_transaction_stats(address: str) -> dict:
    stats : ModelParams = ModelParams()

    # Populate stats with fetched transaction data
    stats.totalEtherBalance = await get_wallet_balance(address)
    transactions = await get_transactions(address)

    received_txns = [tx for tx in transactions if tx["to"].lower() == address.lower()]
    sent_txns = [tx for tx in transactions if tx["from"].lower() == address.lower()]

    # Feature engineering
    stats.receivedTnx = len(received_txns)
    stats.sentTnx = len(sent_txns)
    stats.totalEtherSent = sum(float(tx["value"]) / 1e18 for tx in sent_txns)
    stats.totalEtherReceived = sum(float(tx["value"]) / 1e18 for tx in received_txns)
    if len(transactions) > 1:
        first_tx_time = int(transactions[0]["timeStamp"])
        last_tx_time = int(transactions[-1]["timeStamp"])
        stats.timeDiffFirstLastMins = (last_tx_time - first_tx_time) / 60

    if stats.receivedTnx > 0:
        stats.avgValReceived = stats.totalEtherReceived / stats.receivedTnx

    if stats.sentTnx > 0:
        stats.avgValSent = stats.totalEtherSent / stats.sentTnx

    if stats.receivedTnx > 1:
        received_times = [int(tx["timeStamp"]) for tx in received_txns]
        received_time_diffs = [(received_times[i] - received_times[i - 1]) / 60 for i in range(1, len(received_times))]
        stats.avgMinBetweenReceivedTnx = float(
            sum(received_time_diffs) / len(received_time_diffs)
        ) if len(received_time_diffs) > 0 else 0

    if stats.sentTnx > 1:
        sent_times = [int(tx["timeStamp"]) for tx in sent_txns]
        sent_time_diffs = [(sent_times[i] - sent_times[i - 1]) / 60 for i in range(1, len(sent_times))]
        stats.avgMinBetweenSentTnx = float(
            sum(sent_time_diffs) / len(sent_time_diffs)
        ) if len(sent_time_diffs) > 0 else 0

    stats.totalTransactions = stats.sentTnx + stats.receivedTnx
    stats.ratioRecSent = stats.receivedTnx / stats.sentTnx if stats.sentTnx > 0 else 0
    stats.ratioSentTotal = stats.sentTnx / stats.totalTransactions if stats.totalTransactions > 0 else 0
    stats.ratioRecTotal = stats.receivedTnx / stats.totalTransactions if stats.totalTransactions > 0 else 0

    return stats


@cached(cache=TTLCache(maxsize=1024, ttl=300))
async def process(eth_request : str):
    print("ML_Score for ", eth_request)
    try:
        features = await fetch_transaction_stats(eth_request)

        # Arrange features in expected order for the model
        feature_order = [
            'avgMinBetweenSentTnx', 'avgMinBetweenReceivedTnx', 'timeDiffFirstLastMins',
            'sentTnx', 'receivedTnx', 'avgValSent', 'avgValReceived', 'totalTransactions',
            'totalEtherSent', 'totalEtherReceived', 'totalEtherBalance', 'ratioRecSent',
            'ratioSentTotal', 'ratioRecTotal'
        ]
        feature_values = np.array([getattr(features, key) for key in feature_order]).reshape(1, -1)

    # Perform prediction
        prediction = model.predict(feature_values)
        print("ML Score done for ", eth_request)
        return prediction
    except Exception as e:
        print(f"Error calculating ML score: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error in prediction: {str(e)}")