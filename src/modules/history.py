from pandas import DataFrame
from src.database.setup_db import DatabaseSetup


def _parse(
    cust_info: DataFrame,
    purchase: DataFrame,
    recent_n: int = 5,
) -> str:
    """
    Generates a formatted string containing customer information and their most recent purchase history.

    Args:
        cust_info (DataFrame): A DataFrame containing customer information.
        purchase (DataFrame): A DataFrame containing purchase history for the customer, with at least 'txn_dt' (transaction date) and 'item_name' columns.
        recent_n (int, optional): The number of most recent purchases to include in the output. Defaults to 5.
    
    Returns:
        str: A formatted string with customer details and a list of their most recent purchases.
    """
    cust_template = """# Customer Information
- Gender: {gender}
- Age: {age}
- Occupation: {occupation}
- Address: {address}
"""
    purchase_template = """# Purchase History (The most recent {recent_n} purchases)
- {previous_purchase}
"""
    cust_row = cust_info.iloc[0]
    customer_id = cust_row["customer_id"]
    cust_str = cust_template.format(
        gender=cust_row["gender"],
        age=cust_row["age"],
        occupation=cust_row["occupation"],
        address=cust_row["address"],
    )

    purchase = (
        purchase
        .sort_values(by="txn_dt", ascending=False)
        .head(recent_n)
    )

    previous_purchase = purchase["item_name"].values.tolist()
    previous_purchase = "\n- ".join(previous_purchase)

    purchase_str = purchase_template.format(
        recent_n=recent_n,
        previous_purchase=previous_purchase,
    )

    return cust_str + "\n\n" + purchase_str
    
def get_purchase_history(cust_info: DataFrame):
    """
    Retrieve purchase history for a given customer ID.

    Args:
        customer_id (int): The ID of the customer whose purchase history is to be retrieved.

    Returns:
        DataFrame: A DataFrame containing the customer's purchase history.
    """
    db = DatabaseSetup()
    customer_id = cust_info.iloc[0]["customer_id"]
    sql = db.sql["purchase"].format(customer_id=customer_id)
    purchase = db.query(sql)
    db.close()
    return _parse(cust_info, purchase)


if __name__ == "__main__":
    from src.modules.login import login

    authen_res = login(email="oui@aift.in.th", password=1234)
    cust_info = authen_res["cust_info"]
    history = get_purchase_history(cust_info)
    print(history)
