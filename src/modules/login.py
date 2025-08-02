from typing import Dict, Any
from src.database.setup_db import DatabaseSetup


def login(email: str, password: str) -> Dict[str, Any]:
    """
    Login function to authenticate user credentials.

    Args:
        email (str): User's email.
        password (str): User's password.

    Returns:
        bool: True if login is successful, False otherwise.
    """
    db = DatabaseSetup()
    sql = db.sql["cust_info"].format(email=email)
    result = db.query(sql)
    db.close()
    
    authentication = False
    if not result.empty and result.iloc[0]["password"] == password:
        authentication = True
    
    return {
        "authentication": authentication,
        "cust_info": result,
    }


if __name__ == "__main__":
    # print(login(email="x", password=1234))
    print(login(email="oui@aift.in.th", password=1234))
