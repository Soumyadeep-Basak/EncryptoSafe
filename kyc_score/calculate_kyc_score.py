from fastapi import HTTPException
import utils.fetchEtherscanAPI as fetchEtherscanAPI
from db.mongodb import get_collections
from core.config import Config
from cachetools import cached, TTLCache


@cached(cache=TTLCache(maxsize=1024, ttl=300))
async def kyc_score(eth_address: str) -> float:
    print("KYC score for", eth_address)
    try:
        # Get the KYC collection
        kyc_collection = get_collections(Config.MONGO_URI).get_kyc_collection()

        # Fetch transactions from Etherscan
        response = await fetchEtherscanAPI.fetch_eth_wallet_transactions(
            input_eth_wallet=eth_address
        )

        if response and response.get('status') == '1':
            transactions = response.get('result')

            # Collect all unique addresses from 'from' and 'to'
            all_addresses = {txn.get('from') for txn in transactions} | {txn.get('to') for txn in transactions}

            # Query KYC collection with $in to check if any address exists
            kyc_addresses = set(
                doc['address'] for doc in kyc_collection.find({"address": {"$in": list(all_addresses)}})
            )

            # Count how many transactions involve KYC-ed addresses
            count = sum(1 for txn in transactions if txn.get('from') in kyc_addresses or txn.get('to') in kyc_addresses)

            print(f"KYC score done for {eth_address}")
            return 1.0 if count > 0 else 0.0

    except Exception as e:
        print(f"Error calculating KYC score: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error in KYC score calculation: {str(e)}")
