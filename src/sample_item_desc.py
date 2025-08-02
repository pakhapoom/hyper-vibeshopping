from pandas import read_csv


df = read_csv("data/db/data_with_base64.csv").sample(5)

for _, row in df.iterrows():
    print("- {name}: {description}".format(
        name=row["name"],
        description=row["description"],
    ))
    