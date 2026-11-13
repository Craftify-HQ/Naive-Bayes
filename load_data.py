import pandas as pd

DATA_PATH = "w_review_train.csv"


def load_data(path=DATA_PATH):
    """โหลดข้อมูลรีวิว คืนค่า DataFrame ที่มีคอลัมน์ review, rating"""
    if path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(path, header=None)
    else:
        df = pd.read_csv(path, sep=None, engine="python", header=None)

    df = df.iloc[:, :2]
    df.columns = ["review", "rating"]

    # ตัดแถวหัวตาราง/แถวที่ rating ไม่ใช่ตัวเลขออก
    df = df[pd.to_numeric(df["rating"], errors="coerce").notna()].copy()
    df["rating"] = df["rating"].astype(int)
    df["review"] = df["review"].astype(str)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.head())
    print(df["rating"].value_counts().sort_index())
