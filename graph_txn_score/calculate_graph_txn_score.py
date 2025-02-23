import threading
from datetime import datetime
import aiohttp
import asyncio
from fastapi import HTTPException
import utils.fetchBlacklist as fetchBlacklist
import utils.fetchEtherscanAPI as fetchEtherscanAPI
from cachetools import cached, TTLCache

def calculate_txn_accs(response, input_eth_wallet, unique_address_limit):
    """
    Performs calculations on the API response to extract the necessary transaction counts.
    """
    txn_accs = {}
    if response and response.get("status") == "1":
        transactions = response["result"]
        for txn in transactions:
            from_addr, to_addr = txn["from"], txn["to"]

            if from_addr != input_eth_wallet:
                txn_accs[from_addr] = txn_accs.get(from_addr, 0) + 1
            if to_addr != input_eth_wallet:
                txn_accs[to_addr] = txn_accs.get(to_addr, 0) + 1

            if len(txn_accs) >= unique_address_limit:
                return txn_accs, transactions[-1]["blockNumber"]

        return txn_accs, transactions[-1]["blockNumber"]

    return txn_accs, 99999999


async def helper1(eth_wallet, unique_address_limit):
    """
    Helper function to fetch transactions and calculate results.
    """
    # Step 1: Fetch the transaction data
    response = await fetchEtherscanAPI.fetch_eth_wallet_transactions(
        input_eth_wallet=eth_wallet.lower()
    )
    # Step 2: Perform the calculation on the fetched data
    return calculate_txn_accs(response, eth_wallet, unique_address_limit)


async def calculate_third_lvl_score(acc, blacklist):
    """
    Fetches third-level account transactions and calculates the score.
    """
    third_lvl_acc_limit = 1000
    txn_accs, _ = await helper1(acc, third_lvl_acc_limit)

    blacklist_txn, total_txn = 0, 0
    if txn_accs:
        for acc3, count in txn_accs.items():
            if acc3 in blacklist:
                blacklist_txn += count
            total_txn += count

    return acc, blacklist_txn, total_txn

@cached(cache=TTLCache(maxsize=1024, ttl=300))
async def graph_txn_score(eth_address : str) -> float:
    print("Graph Txn score for", eth_address)
    start_time = datetime.now()
    count = 0
    try:
        blacklist = set(fetchBlacklist.fetchBlacklist()["address"].values)

        if eth_address.lower() in blacklist:
            return 1

        second_lvl_acc_limit = 300

        # Step 1: Fetch second-level accounts and transaction data
        second_lvl_accs, _ = await helper1(eth_address, second_lvl_acc_limit)
        count += 1

        if not second_lvl_accs:
            return 0

        # Step 2: Create a list of tasks for third-level account calculations
        third_lvl_tasks = []
        for acc in second_lvl_accs:
            third_lvl_tasks.append(calculate_third_lvl_score(acc, blacklist))
            count += 1

        # Step 3: Run all third-level tasks concurrently and wait for results
        third_lvl_results = await asyncio.gather(*third_lvl_tasks)

        second_lvl_scores = {acc: 0 for acc in second_lvl_accs}

        # Step 4: Process third-level results
        for acc, blacklist_txn, total_txn in third_lvl_results:
            if total_txn:
                second_lvl_scores[acc] = blacklist_txn / total_txn

        # Step 5: Calculate the overall score
        overall_blacklist_txn = 0
        overall_total_txn = 0

        for acc, txn_count in second_lvl_accs.items():
            if acc in blacklist:
                overall_blacklist_txn += txn_count
                overall_total_txn += txn_count
            else:
                overall_blacklist_txn += second_lvl_scores[acc] * txn_count
                overall_total_txn += txn_count

        score = overall_blacklist_txn / overall_total_txn if overall_total_txn else 0
        print(f"Graph Txn score done for {eth_address}")
        return score ** (1 / 2.5)

    except Exception as e:
        print(f"Error calculating Graph Txn score: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error in Graph Txn Score calculation: {str(e)}")
