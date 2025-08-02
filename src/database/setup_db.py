import sqlite3
from pandas import read_excel, read_sql_query


class DatabaseSetup:

    sql = {
        "cust_info": "SELECT * FROM cust_info WHERE email = '{email}'",
        "purchase": "SELECT * FROM purchase WHERE customer_id = {customer_id}",
    }

    def __init__(self):
        self.setup()

    def load_data(self, data_path: str, table_name: str) -> None:
        data = read_excel(data_path)
        data.to_sql(table_name, self.conn, index=False, if_exists="replace")

    def setup(self):
        self.conn = sqlite3.connect(":memory:")
        self.load_data("data/db/cust_info.xlsx", "cust_info")
        self.load_data("data/db/purchase.xlsx", "purchase")

    def query(self, sql):
        return read_sql_query(sql, self.conn)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db = DatabaseSetup()
    # result = db.query(db.sql["cust_info"].format(email="oui@aift.in.th"))
    result = db.query(db.sql["purchase"].format(customer_id=2))
    print(result)
    db.close()
