# Customer_analysis
# Giới thiệu chủ đề phân tích
Phân tích này sử dụng bộ dữ liệu tài chính kết hợp của tổ chức ngân hàng, trải dài trong suốt thập niên 2010. Với mục tiêu tìm đối tượng khách hàng và nhóm dịch vụ tiềm năng nhằm phát triển sản phầm, tái cấu trúc chính sách hạn mức và kích thích nhu cầu tín dụng. 
# Giới thiệu bộ dữ liệu sử dụng:
Bộ dữ liệu được sử dụng được lấy từ https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets. Đây là bộ dữ liệu tài chính lớn gồm 5 file chính:
- Transaction Data (transactions_data.csv)
- Card Information (cards_data.csv)
- User Data (users_data.csv)
- Merchant Category Codes (mcc_codes.json)
- Fraud Labels (train_fraud_labels.json)

# Chi tiết dự án:
# Tiền xử lý dữ liệu:
* Nhập dữ liệu và thư viện cần thiết:
```
import pandas as pd
import json 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

transactions_data = pd.read_csv('/Users/thekiet/Downloads/Personal_project_data/transactions_data.csv')
cards_data = pd.read_csv('/Users/thekiet/Downloads/Personal_project_data/cards_data.csv')
users_data = pd.read_csv('/Users/thekiet/Downloads/Personal_project_data/users_data.csv')

with open('/Users/thekiet/Downloads/Personal_project_data/mcc_codes.json') as f:
    mcc_data = json.load(f)
with open('/Users/thekiet/Downloads/Personal_project_data/train_fraud_labels.json') as f:
    frauds_data = json.load(f)
mcc_df = pd.DataFrame(list(mcc_data.items()), columns=['MCC_Code', 'Description'])
frauds_df = pd.json_normalize(frauds_data).T.reset_index()
frauds_df.columns = ['index', 'target']
frauds_df['index'] = frauds_df['index'].str.replace('target.', '', regex=False)
```
* Xoá các cột không dùng cho mục đích phân tích
```
transactions_data = transactions_data.drop(columns=[
    'merchant_city',
    'merchant_state',
    'zip',
    'use_chip',
    'errors'])
users_data = users_data.drop(columns=[
    'retirement_age',
    'birth_year',
    'birth_month',
    'per_capita_income',
    'address',
    'latitude',
    'longitude',
    'num_credit_cards'])
cards_data = cards_data.drop(columns=[
    'cvv',
    'expires',
    'card_number',
    'card_brand',
    'year_pin_last_changed',
    'card_on_dark_web',
    'num_cards_issued',
    'acct_open_date'])

#Check lại các dataframe

print(transactions_data.head(5))
print(users_data.head(5))
print(cards_data.head(5))
```
* Kiểm tra NA 
```
print(transactions_data.isna().sum())
print(users_data.isna().sum())
print(cards_data.isna().sum())
print(mcc_codes.isna().sum())
print(frauds_df.isna().sum())
```
* Nhận xét: Tất cả các trường dữ liệu được sử dụng đều không có NA
* Xoá bỏ các giao dịch gian lận thông qua việc đối chiếu mã giao dịch với mã giao dịch được đánh dấu là lửa đảo, đảm bảo dữ liệu có ý nghĩa thống kê
```
frauds_df['index'] = frauds_df['index'].astype(str)
transactions_data['id'] = transactions_data['id'].astype(str)

# Tìm các giao dịch có gian lận để loại bỏ trước khi sử dụng data
non_no_frauds = frauds_df[frauds_df['target'] != 'No']

# Lọc transactions lừa đảo
if len(non_no_frauds) > 0:
    clean_transactions = transactions_data[~transactions_data['id'].isin(non_no_frauds['index'])]
    print(f"Đã loại bỏ {len(transactions_data) - len(clean_transactions)} giao dịch lừa đảo")
else:
    clean_transactions = transactions_data.copy()
    print("Không có giao dịch lừa đảo")
```
Chuẩn hoá kiểu dữ liệu cho các biến mang giá trị ngày tháng và tiền tệ (giá trị giao dịch, thu nhập hàng năm, tổng nợ, hạn mức tín dụng) thành dạng số
```
clean_transactions['date'] = pd.to_datetime(clean_transactions['date'])
      # convert to datetime
clean_transactions['amount'] = (
    clean_transactions['amount']
    .str.replace('$', '', regex=False)  # Xóa ký hiệu $
    .str.replace(',', '', regex=False)   # Xóa dấu phân cách nghìn (nếu có)
    .apply(pd.to_numeric)
)

users_data['yearly_income'] = (
    users_data['yearly_income']
    .astype(str)
    .str.replace(r'[\$,]', '', regex=True)
    .str.strip() 
    .apply(pd.to_numeric)
)

users_data['total_debt'] = (
    users_data['total_debt']
    .astype(str)
    .str.replace(r'[\$,]', '', regex=True)
    .str.strip() 
    .apply(pd.to_numeric)
)

cards_data['credit_limit'] = (
    cards_data['credit_limit']
    .astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .apply(pd.to_numeric)
)
```
* Phân loại các mặt hàng vào các nhóm, tạo biến phân loại mặt hàng mới. Mục đích của việc này nhằm loại bỏ các biến phân loại có quá ít giá trị đếm, tổng hợp lại chúng theo nhóm chung hơn, thuận tiện cho phân tích sau
```
# Tạo map gán tên ngành hàng vào mỗi giao dịch nhằm phân loại giao dịch theo ngành hàng
mcc_map = dict(zip(mcc_codes['MCC_Code'].astype(str), mcc_codes['Description'])) # tạo từ điển đối chiếu mã -> tên
clean_transactions['mcc'] = clean_transactions['mcc'].astype(str) #chuyển đổi mã mcc là text
clean_transactions['Description'] = clean_transactions['mcc'].map(mcc_map) 

# Tạo nhóm category lớn, loại outliner category 
def categorize_into_groups(description):
    if pd.isna(description):
        return 'Other'
    
    description_lower = description.lower()
    
    # 1. Ăn uống & Nhà hàng
    if any(keyword in description_lower for keyword in ['restaurant', 'eating', 'food', 'fast food', 'drinking', 'bar', 'cafe']):
        return 'Food & Dining'
    
    # 2. Mua sắm & Bán lẻ
    elif any(keyword in description_lower for keyword in ['store', 'shop', 'department', 'discount', 'wholesale', 'retail', 'clothing', 'shoe', 'cosmetic', 'furniture', 'electronics', 'hardware', 'sporting', 'appliance', 'gift', 'music', 'book', 'florist']):
        return 'Shopping & Retail'
    
    # 3. Dịch vụ cá nhân & Sức khỏe
    elif any(keyword in description_lower for keyword in ['medical', 'doctor', 'dentist', 'hospital', 'chiropractor', 'podiatrist', 'beauty', 'barber', 'laundry', 'cleaning', 'legal', 'insurance', 'accounting', 'tax']):
        return 'Personal Services & Healthcare'
    
    # 4. Du lịch & Giải trí
    elif any(keyword in description_lower for keyword in ['travel', 'hotel', 'motel', 'resort', 'lodging', 'airline', 'cruise', 'railroad', 'railway', 'bus', 'transportation', 'amusement', 'park', 'theater', 'sports', 'recreational', 'betting', 'casino']):
        return 'Travel & Entertainment'
    
    # 5. Xăng dầu & Di chuyển
    elif any(keyword in description_lower for keyword in ['service station', 'gas', 'fuel', 'toll', 'bridge', 'trucking', 'freight', 'towing', 'automotive', 'car wash']):
        return 'Fuel & Transportation'
    
    # 6. Tiện ích & Dịch vụ gia đình
    elif any(keyword in description_lower for keyword in ['utility', 'electric', 'gas', 'water', 'sanitary', 'telecommunication', 'cable', 'satellite', 'tv', 'heating', 'plumbing', 'air conditioning']):
        return 'Utilities & Home Services'
    
    # 7. Mặt hàng công nghệ
    elif any(keyword in description_lower for keyword in ['computer', 'network', 'digital', 'electronic', 'semiconductor', 'software', 'app', 'game']):
        return 'Technology & Digital Goods'
    
    # 8. Vật liệu & Xây dựng
    elif any(keyword in description_lower for keyword in ['lumber', 'building', 'material', 'metal', 'steel', 'iron', 'welding', 'fabrication', 'machinery', 'industrial', 'tool']):
        return 'Construction & Industrial'
    
    # 9. Dịch vụ tài chính
    elif any(keyword in description_lower for keyword in ['money transfer', 'bank', 'financial', 'payment']):
        return 'Financial Services'
    
    # 10. Khác
    else:
        return 'Other'

# Áp dụng phân loại mới cho các mặt hàng 
clean_transactions['category_group'] = clean_transactions['Description'].apply(categorize_into_groups)

print("Số lượng các mặt hàng theo phân loại mới:")
print(clean_transactions['category_group'].value_counts())

```
* Sau đó xử lý các biến dữ liệu với mục đích tạo một bảng kết nối các biến cần thiết để EDA ở các bộ dữ liệu khác nhau thông qua các biến chung (client_id, id,...)
* Thêm biến 'year' giảm số quan sát trong bảng transactions, 'spend' lọc ra các giao dịch tiền ra, có ý nghĩa với việc phân tích thói quen tiêu dùng.
```
# Tạo cột năm quan sát thay đổi
clean_transactions['year'] = clean_transactions['date'].dt.year

# Tạo cột tổng chi tiêu, loại ra các giao dịch tiền vào:
clean_transactions['spend'] = 0
clean_transactions['is_spend'] = clean_transactions['amount'] < 0

# Chỉ gán giá trị cho những dòng là chi tiêu
clean_transactions.loc[clean_transactions['is_spend'], 'spend'] = clean_transactions['amount'].abs()

# Kiểm tra lại độ dài dữ liệu chi tiêu 
len(clean_transactions['spend'])
```
* Nhóm các dữ liệu theo từng khách hàng, lần lượt trong vòng 9 năm. Từ đó chỉ ra mỗi khách hàng chi bao nhiêu tiền, nợ bao nhiêu, hạn mức tín dụng, tổng giao dịch trong mỗi năm, họ chi nhiều tiền nhất từng năm cho ngành hàng nào và chủ yếu những giao dịch trong năm đó dùng loại thẻ nào.
```
# Nhóm dữ liệu, mỗi khách hàng mỗi năm chi bao nhiêu, giao dịch bao nhiêu lần

agg_client_year = clean_transactions.groupby(['client_id', 'year']).agg(
    total_spent=('spend', 'sum'),  # tính tổng chi tiêu
    total_transactions=('id', 'count'),  # số lượng giao dịch
    group_spent=('spend', 'sum')
).reset_index()

#Chia tổng chi tiêu thành nhóm, chủ yếu (0-10k)
spent_bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000, 20000, float('inf')]
labels = ['0-500', '500-1,000', '1,000-1,500', '1,500-2,000', '2,000-2,500', '2,500-3,000', '3,000-5,000', '5,000-10,000', '10,000-20,000', '20,000+']
agg_client_year['group_spent'] = pd.cut(
    agg_client_year['group_spent'], 
    bins=spent_bins, 
    labels=labels, 
    right=False
)
```
```
# Tìm xem mỗi khách hàng chi nhiều tiền nhất cho ngành hàng nào
cat_sum = (
    clean_transactions[clean_transactions['is_spend']]  # Chỉ lấy giao dịch chi tiêu
    .groupby(['client_id', 'year', 'category_group'])  # Nhóm theo khách hàng, năm, nhóm ngành hàng
    .agg(category_spend=('spend', 'sum'))  # Tính tổng chi cho mỗi ngành
    .reset_index()
)

# Sắp xếp để tìm nhóm ngành hàng chi nhiều nhất
cat_sum = cat_sum.sort_values(['client_id', 'year', 'category_spend'], ascending=[True, True, False])

# Lấy nhóm ngành hàng chi nhiều nhất của mỗi khách hàng
top_cat = cat_sum.groupby(['client_id', 'year']).first().reset_index()[['client_id', 'year', 'category_group']]
top_cat = top_cat.rename(columns={'category_group': 'top_category'})
```
```
# Tìm xem khách hàng dùng loại thẻ nào nhiều nhất (credit/debit)
type_cards = clean_transactions.merge(cards_data[['id', 'card_type', 'credit_limit']], left_on='card_id', right_on='id', how='left')

card_group = (
    type_cards[type_cards['is_spend']]  # Chỉ xét chi tiêu
    .groupby(['client_id', 'year', 'card_type'])  # Nhóm theo khách hàng, năm, loại thẻ
    .agg(card_spend=('spend', 'sum'))  # Tổng chi tiêu bằng mỗi loại thẻ
    .reset_index()
)

# Sắp xếp để tìm thẻ được dùng nhiều nhất
card_group = card_group.sort_values(['client_id', 'year', 'card_spend'], ascending=[True, True, False])

# Lấy thẻ chính của mỗi khách hàng
primary_card = card_group.groupby(['client_id', 'year']).first().reset_index()[['client_id', 'year', 'card_type']]

# Lấy credit_limit trung bình cho mỗi client 
client_credit_limit = type_cards.groupby('client_id')['credit_limit'].mean().reset_index()
print(client_credit_limit)

# Lấy tổng nợ khách hàng
client_total_debt = users_data[['id', 'total_debt']].rename(columns={'id': 'client_id'})
print(client_total_debt.head())
```
* Gộp thành 1 bảng
```
# Kết hợp tất cả thông tin thành 1 bảng duy nhất
client_df = (
    agg_client_year  # Tổng chi tiêu
    .merge(top_cat, on=['client_id', 'year'], how='left')  # Thêm ngành hàng yêu thích
    .merge(primary_card, on=['client_id', 'year'], how='left')  # Thêm loại thẻ chính
    .merge(
        users_data[['id', 'gender', 'yearly_income', 'credit_score', 'current_age']]
        .rename(columns={'id': 'client_id'}),
        on='client_id',
        how='left'
    )
    .merge(client_credit_limit, on='client_id', how='left')  
    .merge(client_total_debt, on='client_id', how='left')
)
print(client_df.head())
```
* Chọn từ bảng ra các cột cần phân tích, lưu vào một bảng mới.
```
# Phân tuổi khách hàng thành các nhóm
age_bins = [0, 24, 34, 44, 54, 64, 190]
labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
client_df['age_group'] = pd.cut(client_df['current_age'], bins=age_bins, labels=labels, right=True)

# Phân nhóm thu nhập    
income_bins = [0, 25000, 50000, 75000, 100000, 150000, np.inf]
income_labels = [
    '<25000', 
    '25000–50000', 
    '50000–75000', 
    '75000–100000', 
    '100000–150000', 
    '150000+'
]
client_df['group_income'] = pd.cut(client_df['yearly_income'], bins=income_bins, labels=income_labels, right=False)

# Chọn các cột quan trọng cho phân tích
final_client_df = client_df[[
    'client_id', 'gender', 'age_group', 'yearly_income', 'credit_score', 'credit_limit', 'group_income', 'total_debt',
    'card_type', 'total_spent', 'total_transactions', 'year', 'top_category', 'group_spent'
]]
print(final_client_df.head())
```
* Ở phần này, tiếp tục gộp các cột có số lượng biến lớn như tuổi, thu nhập, chi tiêu, làm gọn bộ dữ liệu nhưng vẫn giữ các cột gốc trong bảng client_df, chỉ chọn lấy những biến đã được gộp theo nhóm vào final_client_df để thuận tiện cho EDA.
* Kiểm tra loại dữ liệu trong các cột và phân loại vào 2 nhóm 'Không đổi qua các năm' và 'Thay đổi qua các năm' để tạo 2 bảng dữ liệu phụ của final_client_df nhằm tránh trùng lặp biến khi vẽ các biểu đồ quan hệ
```
# Kiểm tra phân loại biến static(giống nhau qua các năm), dynamic(có sự khác biệt qua các năm)

columns_to_check = [col for col in final_client_df.columns if col not in ['client_id', 'year']]
static_columns = []
dynamic_columns = []

for column in columns_to_check:
    unique_counts = final_client_df.groupby('client_id')[column].nunique()
    if (unique_counts == 1).all():
        static_columns.append(column)
        print(f" {column}: STATIC")
    else:
        dynamic_columns.append(column)
        print(f" {column}: DYNAMIC")

#Kiểm tra phân loại biến
print(f"Biến ở bảng static: {static_columns}")
print(f"Biến ở bảng dynamic: {dynamic_columns}")

# Chia biến vào 2 bảng dữ liệu tránh trùng lặp dữ liệu khi vẽ biểu đồ quan hệ
client_static_info = final_client_df[['client_id'] + static_columns].drop_duplicates()
client_dynamic_info = final_client_df[['client_id', 'year'] + dynamic_columns]

# Kiểm tra độ dài dữ liệu sau khi chia bảng
print(f"Bảng static info: {client_static_info.shape}")
print(f"Bảng dynamic info: {client_dynamic_info.shape}")
print(f"Số khách hàng unique - Static: {client_static_info['client_id'].nunique()}")
print(f"Số khách hàng unique - Dynamic: {client_dynamic_info['client_id'].nunique()}")
```
# EDA
* Phân tích phân bố các biến trong toàn bộ khung thời gian (2010-2019) dựa vào 3 bảng đã được xử lý: 'final_client_df', 'client_static_info', 'client_dynamic_info'
```
plt.figure(figsize=(20, 15))

# Phân bố độ tuổi - dùng client_static_info
plt.subplot(3, 3, 1)
age_dist = client_static_info['age_group'].value_counts()
plt.pie(age_dist.values, labels=age_dist.index, autopct='%1.1f%%', startangle=90)
plt.title('Phân bố độ tuổi khách hàng')

# Phân bố giới tính - dùng client_static_info
plt.subplot(3, 3, 2)
gender_dist = client_static_info['gender'].value_counts()
plt.pie(gender_dist.values, labels=gender_dist.index, autopct='%1.1f%%', startangle=90)
plt.title('Phân bố giới tính')

# Phân bố thu nhập - dùng client_static_info
plt.subplot(3, 3, 3)
income_dist = client_static_info['group_income'].value_counts()
plt.bar(range(len(income_dist)), income_dist.values)
plt.title('Phân bố thu nhập hàng năm ($)')
plt.xticks(range(len(income_dist)), income_dist.index, rotation=45)
plt.ylabel('Số lượng khách hàng')

# Phân bố điểm tín dụng - dùng client_static_info
plt.subplot(3, 3, 4)
plt.hist(client_static_info['credit_score'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Phân bố điểm tín dụng')
plt.xlabel('Điểm tín dụng')
plt.ylabel('Số lượng')

# Phân bố loại thẻ tất cả khách hàng dùng trong 9 năm 
plt.subplot(3, 3, 5)
card_dist = final_client_df['card_type'].value_counts()
plt.pie(card_dist.values, labels=card_dist.index, autopct='%1.1f%%', startangle=90)
plt.title('Phân bố loại thẻ')

# Tần suất thực hiện giao dịch của các khách hàng mỗi năm trong giai đoạn 2010-2019 - dùng final_client_df
plt.subplot(3, 3, 6)
plt.hist(final_client_df['total_transactions'], bins=30, edgecolor='black')
plt.title('Phân bố số lượng giao dịch')
plt.xlabel('Số giao dịch')
plt.ylabel('Tần suất')

#Phân bố hạn mức tín dụng các khách hàng mỗi năm trong giai đoạn 2010-2019
plt.subplot(3, 3, 7)
plt.hist(final_client_df['credit_limit'], bins=35, edgecolor='black')
plt.title('Phân bố hạn mức tín dụng')
plt.xlabel('Hạn mức tín dụng ($)')
plt.ylabel('Tần suất')

#Phân bố hạn mặt hàng chi tiêu các khách hàng mỗi năm trong giai đoạn 2010-2019
plt.subplot(3, 3, 8)
category_counts = client_dynamic_info['top_category'].value_counts()
plt.bar(range(len(category_counts)), category_counts.values)
plt.title('Phân bố mặt hàng chi tiêu')
plt.xticks(range(len(category_counts)), category_counts.index, rotation=45)
plt.ylabel('Số giao dịch')

# Phân bố tổng nợ 
plt.subplot(3, 3, 9)
plt.hist(client_static_info['total_debt'], bins=35, edgecolor='black')
plt.title('Phân bố tổng nợ')
plt.xlabel('Tổng nợ ($)')
plt.ylabel('Tần suất')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()
```
![i5](https://i.imgur.com/6efB45m.png)














