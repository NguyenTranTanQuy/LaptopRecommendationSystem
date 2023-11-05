import os
import numpy as np
import pandas as pd

from pyvi import ViTokenizer, ViPosTagger
from sklearn.preprocessing import MinMaxScaler

current_file = __file__
f = os.path.dirname((os.path.abspath(current_file)))

# Get stopwords of Vietnamese
with open(f + '/stopwords/vietnamese-stopwords.txt', 'r', encoding="utf-8") as file:
    stop_words = file.read().splitlines()


# Define a function to preprocess text
def preprocess_text(text):
    tokens = ViPosTagger.postagging(ViTokenizer.tokenize(text))[0]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def preprocess_data():
    # Read a excel file
    data = pd.read_csv(f + "/datasets/Laptops.csv", encoding="utf-8")

    # Remove duplicated Data
    data = data.drop_duplicates(subset='Features', ignore_index=False)
    data = data.drop_duplicates(subset='Name', ignore_index=False)
    data = data.dropna(subset='Features', ignore_index=False)
    data = data.reset_index(drop=True)

    # Data Reduction: Drop some columns
    data = data.drop(['Loại RAM', 'Tần số quét', 'Bảo mật', 'Loại màn hình',
                      'Độ phân giải màn hình', 'Công nghệ màn hình', 'Tính năng đặc biệt',
                      'Wi-Fi', 'Công nghệ âm thanh', 'Unnamed: 31',
                      'Chất liệu tấm nền', 'Socket', 'Kích thước',
                      'Độ phân giải', 'Số khe ram',
                      'Loại đèn bàn phím', 'Cổng giao tiếp', 'Khe đọc thẻ nhớ',
                      'Rating', 'Numbers of Rating'], axis=1)

    # Handle Values:
    data['Dung lượng RAM'] = data['Dung lượng RAM'].fillna(data['Dung lượng RAM'].mode()[0])
    data['Dung lượng RAM'] = data['Dung lượng RAM'].str.replace('GB', '')
    data['Loại card đồ họa'] = data['Loại card đồ họa'].apply(lambda x: 1 if pd.notnull(x) and x != '' else 0)
    data['Chất liệu'] = data['Chất liệu'].fillna(data['Chất liệu'].mode()[0])
    data['Hãng sản xuất'] = data['Hãng sản xuất'].fillna(data['Name'].apply(lambda x: x.split()[1].upper() if len(x.split()) > 1 else x.upper()))
    data['Webcam'] = data['Webcam'].apply(lambda x: 1 if x == 'Có' else 0)
    data['Màn hình cảm ứng'] = data['Màn hình cảm ứng'].apply(lambda x: 1 if x == 'Có' else 0)
    data['Hệ điều hành'] = data['Hệ điều hành'].fillna(data['Hệ điều hành'].mode()[0])
    data['Kích thước màn hình'] = data['Kích thước màn hình'].str.replace('inches', '')
    data['Bluetooth'] = data['Bluetooth'].apply((lambda x: 1 if pd.notnull(x) and x != '' else 0))
    data['Pin'] = data['Pin'].fillna(data['Pin'].mode()[0])
    data['Trọng lượng'] = data['Trọng lượng'].fillna(data['Trọng lượng'].mode()[0])

    # Assuming df is your DataFrame and 'Pin' is the name of the column
    data['Pin'] = data['Pin'].str.extract('(\d+\.?\d*)\s?(?:W|WH|Wh)')
    data['Pin'] = pd.to_numeric(data['Pin'])
    data['Pin'] = data['Pin'].fillna(round(data['Pin'].mean()))

    data['Trọng lượng'] = data['Trọng lượng'].str.lower()
    data['Trọng lượng'] = data['Trọng lượng'].str.replace(',', '.')
    data['Trọng lượng'] = data['Trọng lượng'].str.lower().str.extract('(\d+\.?\d*)\s?(?:g|gram|kg)')
    data['Trọng lượng'] = pd.to_numeric(data['Trọng lượng'])
    data['Trọng lượng'] = data['Trọng lượng'].apply(lambda x: x if x < 10 else x / 1000)
    data['Trọng lượng'] = data['Trọng lượng'].fillna(round(data['Trọng lượng'].mean(), 2))

    # Data Transformation: I will scale the numerical columns of Laptops to a range of 0-1
    scaler = MinMaxScaler()

    data['Numbers of Comment'] = scaler.fit_transform((data[['Numbers of Comment']]))
    data['Dung lượng RAM'] = scaler.fit_transform((data[['Dung lượng RAM']]))
    data['Kích thước màn hình'] = scaler.fit_transform((data[['Kích thước màn hình']]))
    data['Pin'] = scaler.fit_transform((data[['Pin']]))
    data['Trọng lượng'] = scaler.fit_transform((data[['Trọng lượng']]))

    # Preprocess columns
    for i in range(len(data['Features'])):
        try:
            feature = preprocess_text(data.loc[i]['Features'].lower())
            data.loc[i, 'Features'] = " ".join(feature)
            data.loc[i, 'Features'] = data.loc[i, 'Features'].replace('\n', '')
            data.loc[i, 'Features'] = data.loc[i, 'Features'].replace('.', '')
            data.loc[i, 'Features'] = data.loc[i, 'Features'].replace('  ', ' ')
        except:
            continue

    # Choose some important columns to save DataFrame back to the csv file
    data.to_csv(f + '/datasets/result.csv', encoding="utf-8", index=False, columns=['Nhãn', 'Features'])
