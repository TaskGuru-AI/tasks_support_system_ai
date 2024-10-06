import pandas as pd


def test_pandas():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    print("Sample DataFrame:")
    print(df)

    df["C"] = df["A"] + df["B"]

    print("\nUpdated DataFrame:")
    print(df)

    print("\nPandas is installed and running correctly!")


if __name__ == "__main__":
    test_pandas()
