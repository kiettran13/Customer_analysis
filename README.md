# Phân tích hành vi khách hàng
# Mục tiêu dự án
Với mục tiêu tìm đối tượng khách hàng và nhóm dịch vụ tiềm năng nhằm phát triển sản phầm và kích thích nhu cầu tín dụng.
# Tổng quan dự án 
Dự án tập trung phân tích hành vi giao dịch của khách hàng dựa trên đặc điểm nhân khẩu học và mô hình chi tiêu thực tế, đồng thời đánh giá động lực sử dụng Credit Card theo độ tuổi, thu nhập và điểm tín dụng. Từ đó nhận diện những nhóm khách hàng có mức độ tiềm năng cao, hướng đến việc xác định đối tượng phù hợp nhất cho các sản phẩm tín dụng.
# Giới thiệu bộ dữ liệu sử dụng:
Bộ dữ liệu được sử dụng được lấy từ [Kaggle](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets). Đây là bộ dữ liệu tài chính lớn gồm 5 file chính:
- Transaction Data (transactions_data.csv)
- Card Information (cards_data.csv)
- User Data (users_data.csv)
- Merchant Category Codes (mcc_codes.json)
- Fraud Labels (train_fraud_labels.json)

Bộ dữ liệu được chia sẻ ở trong folder [data](https://github.com/kiettran13/Customer_analysis/tree/964c99f7904bc7fa2fb1dace7d3f2466fff7a94f/data). Riêng dữ liệu giao dịch (Transaction Data) và dữ liệu về dấu hiệu lừa đảo (Fraud Labels) có kích thước lớn nên sẽ được chia sẻ qua link.

# Chi tiết dự án:
# Tiền xử lý dữ liệu:
* Nhập dữ liệu và thư viện cần thiết:

<details>
<summary><b>Click để xem toàn bộ code </b></summary>
    
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
</details>

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
*Nhận xét:* Tất cả các trường dữ liệu được sử dụng đều không có NA
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

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
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
</details>

* Phân loại các mặt hàng vào các nhóm, tạo biến phân loại mặt hàng mới. Mục đích của việc này nhằm loại bỏ các biến phân loại có quá ít giá trị đếm, tổng hợp lại chúng theo nhóm chung hơn, thuận tiện cho phân tích sau

<details>
<summary><b>Xem toàn bộ code </b></summary>

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
</details>

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

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
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
</details>

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
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
</details>
    
<details>
<summary><b>Click để xem toàn bộ code </b></summary>
    
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
</details>

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

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
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
</details>
    
# EDA
* Ở bước này, tôi sẽ trực quan hoá dữ liệu để khám phá tổng quan tính chất các biến và một số quan hệ cơ bản của biến dựa vào 3 bảng đã được xử lý: 'final_client_df', 'client_static_info', 'client_dynamic_info'
## Phân bố các biến trong toàn bộ khung thời gian (2010-2019 )

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
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
</details>

![i1](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/EDA_variables.png)
*Insight:*
- Phân bố nhân khẩu học của các khách hàng khá đồng đều giữa các nhóm, phân bố độ tuổi giao động trong khoảng 5% giữa các nhóm, khách hàng 45-54 chiếm tỷ lệ lớn nhất (25,3%), khách hàng nhóm tuổi dưới 25 chiếm tỷ lệ nhỏ nhất, chỉ 0,3%
- Phân bố thu nhập hàng năm phổ biến nhất ở mức 25k-50k $ và 50k-75k $, đây là mức trung bình trong phổ dữ liệu, khách hàng của tổ chức ngân hàng chủ yếu là phân lớp bình dân đến trung lưu.
- Phân bố loại thẻ và số giao dịch và mặt hàng chi tiêu cho thấy, có phần lớn khách hàng giao dịch thường xuyên xung mức trung bình 1000 lần giao dịch mỗi năm trong hầu hết thời gian, và loại thẻ họ chủ yếu sử dụng trong giao dịch là Debit và Credit với tỷ lệ khá cân đối, 55,5% cho Debit và 41,4% cho Credit và top 3 nhóm dịch vụ được khách hành sử dụng nhiều nhất là Thực phẩm & Ăn uống, Xăng dầu & Di chuyển, Du lịch $ Giải trí.
- Khách hàng có sự cẩn trọng trong xu hướng sử dụng nợ và chi tiêu nợ khi hạn mức tín dụng luôn ở mức thấp, trung bình dưới 20k $ cho mỗi năm và điểm tín dụng luôn cao hơn mức trung bình (>650) và một tổng nợ trung bình 1 khách hàng đang gánh chịu chỉ tập trung dưới 10k $ mỗi năm trong thời gian một thập kỷ.

## Phân tích chi tiêu và thu nhập trong mối quan hệ với nhóm tuổi và loại thẻ sử dụng các biến trong toàn bộ khung thời gian

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
```
plt.figure(figsize=(12, 8))

#Heatmap mối quan hệ chi tiêu trung bình theo loại thẻ và nhóm tuổi
plt.subplot(2, 2, 1)
card_spent = final_client_df.groupby(['card_type', 'age_group'])['total_spent'].mean().unstack()
sns.heatmap(card_spent, annot=True, fmt='.0f',cmap='YlGnBu', 
            linewidths=0.5, linecolor='gray')
plt.title('Tổng chi tiêu trung bình theo loại thẻ và nhóm tuổi')
plt.xlabel('Nhóm tuổi')
plt.ylabel('Loại thẻ')
#plt.tight_layout()
#plt.show()

#Heatmap mối quan hệ thu nhập trung bình theo loại thẻ và nhóm tuổi
plt.subplot(2, 2, 2)
card_income = final_client_df.groupby(['card_type', 'age_group'])['yearly_income'].mean().unstack()
sns.heatmap(card_income, annot=True, fmt='.0f',cmap='YlGnBu', 
            linewidths=0.5, linecolor='gray')
plt.title('Tổng thu nhập trung bình theo loại thẻ và nhóm tuổi')
plt.xlabel('Nhóm tuổi')
plt.ylabel('Loại thẻ')

plt.tight_layout()
plt.show()
```
</details>

![i2](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/EDA_heatmap.png)
*Insight:*
- Nhìn vào chi tiêu trung bình có thể thấy những khách hàng sử dụng thẻ Debit trong nhóm tuổi 35-54 cho thấy sự phóng khoáng trong chi tiêu hơn các nhóm tuổi khác trong phân tích, còn những khách hàng sử dụng thẻ Credit lại chi tiêu tương đối đồng đều giữa các nhóm tuổi, chỉ có những người dưới 25 tuổi ở nhóm này có xu hướng chi tiêu mạnh
- Tuy nhiên, ở khía cạnh thu nhập trung bình, những khách hàng dùng thẻ Credit nhìn chung có thu nhập đều và cao hơn nhóm sử dụng thẻ Debit ở mọi nhóm tuổi. Tuy nhiên khác biệt về thu nhập này tương đối thấp.
- Điều này phản ánh sự khác nhau trong thói quen chi tiêu của khách hàng sử dụng 2 loại thẻ mặc dù cả 2 nhóm này đều có mức thu nhập gần như tương đương nhau.

## Phân tích các biến theo sự thay đổi của thời gian

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
```
plt.figure(figsize=(18, 12))

# Số giao dịch trung bình theo năm 
plt.subplot(2, 3, 1)
yearly_transactions = client_dynamic_info.groupby('year')['total_transactions'].mean()
plt.plot(yearly_transactions.index, yearly_transactions.values, marker='o', color='blue')
plt.title('Số giao dịch trung bình qua các năm')
plt.xlabel('Năm')
plt.ylabel('Số giao dịch trung bình')
plt.grid(True)


# Điểm tín dụng trung bình theo năm
yearly_credit_merged = final_client_df.groupby('year')['credit_score'].mean()
plt.subplot(2, 3, 2)
plt.plot(yearly_credit_merged.index, yearly_credit_merged.values, marker='s', color='red')
plt.title('Điểm tín dụng trung bình qua các năm')
plt.xlabel('Năm')
plt.ylabel('Điểm tín dụng trung bình')
plt.grid(True)

# Tổng nợ trung bình theo năm 
plt.subplot(2, 3, 3)
yearly_total_debt = final_client_df.groupby('year')['total_debt'].mean()
plt.plot(yearly_total_debt.index, yearly_total_debt.values, marker='^', color='purple', linewidth=2)
plt.title('Tổng nợ trung bình qua các năm')
plt.xlabel('Năm')
plt.ylabel('Tổng nợ trung bình ($)')
plt.grid(True)

# Phân bố chi tiêu theo năm 
plt.subplot(2, 3, 4)
yearly_group_spent = pd.crosstab(client_dynamic_info['year'], client_dynamic_info['group_spent'])
sns.heatmap(yearly_group_spent, annot=True, cmap='YlOrRd', fmt='d')
plt.title('Phân bố chi tiêu theo năm')
plt.xlabel('Chi tiêu theo nhóm')
plt.ylabel('Năm')

# Phân bố thu nhập theo năm 
plt.subplot(2, 3, 5)
yearly_group_income = pd.crosstab(client_dynamic_info['year'], client_static_info['group_income'])
sns.heatmap(yearly_group_spent, annot=True, cmap='YlGn', fmt='d')
plt.title('Phân bố thu nhập theo năm')
plt.xlabel('Thu nhập theo nhóm')
plt.ylabel('Năm')

# Xu hướng hạn mức tín dụng theo năm
yearly_credit_limit = final_client_df.groupby('year')['credit_limit'].mean()
plt.subplot(2, 3, 6)
plt.plot(yearly_credit_limit.index, yearly_credit_limit.values, marker='s', color='blue')
plt.title('Hạn mức tính dụng trung bình qua các năm')
plt.xlabel('Năm')
plt.ylabel('Hạn mức tín dụng trung bình')
plt.grid(True)

plt.tight_layout()
plt.show()
```
</details>
    
![i3](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/EDA_time.png)
*Insight:*
- Hầu hết trong bộ dữ liệu đều có thay đổi qua các năm, tuy nhiên các thay đổi này hoàn toàn không đáng kể. Điều này cho thấy mức độ ổn định từ tệp khách hàng của tổ chức tài chính.

## Tổng kết phân tích EDA
Có thể thấy tổ chức tài chính này có một tệp khách hàng ổn định với hành vi khách hàng thay đổi tương đối ít qua các năm. Đặc trưng bởi các biến về nhân khẩu học và các biến về tài chính cá nhân và hành vi tài chính. Với một tệp khách hàng chắc chắn và thói quen lâu năm, khách hàng có sự trung thành và gắn kết cao với dịch vụ của tổ chức ngân hàng. Thông qua EDA tổng quát có thể thấy, tệp khách hàng ở nhóm tuổi 35-54 đang là tệp khách hàng đông đảo, chiếm đến hơn 46% khách hàng, đây cũng là nhóm khách hàng đang có chi tiêu cao nhất ở nhóm thẻ Debit, tuy nhiên lại chưa có sự đột biến trong chi tiêu ở thẻ tín dụng, mặc dù họ có mức tương đối cao so với chi tiêu hàng năm.

# Phân tích sâu hơn hành vi nhóm khách hàng sử dụng thẻ tín dụng 
* Ở phần này, tôi lọc những khách hàng sử dụng thẻ tín dụng (Credit) và sử dụng heatmap và subplot phân tích so sánh các biến hạn mức, tần suất giao dịch và tổng nợ giữa nhóm sử dụng thẻ tín dụng và tổng thể khách hàng, từ đó xem xét tính chất, thói quen nhóm khách hàng sử dụng thẻ tín dụng
## Phân bố loại thẻ theo nhóm tuổi 

<details>
<summary><b>Xem toàn bộ code </b></summary>

```
plt.figure(figsize=(12, 6))
card_age = pd.crosstab(final_client_df['age_group'], final_client_df['card_type'])
card_age.plot(kind='bar', stacked=True)
plt.title('Phân bố loại thẻ theo độ tuổi')
plt.xlabel('Độ tuổi')
plt.ylabel('Số lượng')
plt.legend(title='Loại thẻ')
plt.show()
```
</details>

## Phân bố chi tiêu & thu nhập theo nhóm tuổi và loại thẻ

<details>
<summary><b>Xem toàn bộ code </b></summary>

```
# Tạo dataset cho credit card users
credit_card_users = final_client_df[final_client_df['card_type'] == 'Credit']
50
# Tạo heatmap
plt.figure(figsize=(20, 15))
# 1. SO SÁNH PHÂN BỐ ĐỘ TUỔI VS THU NHẬP, XEM SỰ KHÁC NHAU CỦA NHÂN KHẨU HỌC

# Tổng thể
plt.subplot(2, 2, 1)
age_income_all = pd.crosstab(final_client_df['age_group'], final_client_df['group_income'])
sns.heatmap(age_income_all, annot=True, cmap='Blues', fmt='d')
plt.title('Phân bố độ tuổi và thu nhập\n(Tổng thể)', fontsize=14, fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Độ tuổi')

# Chỉ credit card users
plt.subplot(2, 2, 2)
age_income_credit = pd.crosstab(credit_card_users['age_group'], credit_card_users['group_income'])
sns.heatmap(age_income_credit, annot=True, cmap='Blues', fmt='d')
plt.title('Phân bố độ tuổi và thu nhập\n(Chỉ khách hàng sử dụng thẻ tín dụng)', fontsize=14, fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Độ tuổi')

# 2. SO SÁNH ĐIỂM TÍN DỤNG THEO ĐỘ TUỔI VÀ THU NHẬP

# Tổng thể
plt.subplot(2, 2, 3)
age_income_credit_limit_all = final_client_df.groupby(['age_group', 'group_income'])['credit_limit'].mean().unstack()
sns.heatmap(age_income_credit_limit_all, annot=True, cmap='RdYlBu', fmt='.0f', center=700)
plt.title('Hạn mức tín dụng trung bình trong các năm\n(Tổng thể)', fontsize=14, fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Độ tuổi')

# Chỉ credit card users
plt.subplot(2, 2, 4)
age_income_credit_limit_only = credit_card_users.groupby(['age_group', 'group_income'])['credit_limit'].mean().unstack()
sns.heatmap(age_income_credit_limit_only, annot=True, cmap='RdYlBu', fmt='.0f', center=700)
plt.title('Hạn mức tín dụng trung bình trong các năm\n(Chỉ khách hàng sử dụng thẻ tín dụng )', fontsize=14, fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Độ tuổi')

plt.tight_layout()
plt.show()
```
```
# 3. SO SÁNH CHI TIÊU VÀ GIAO DỊCH THEO ĐỘ TUỔI
plt.figure(figsize=(20, 15))

# Số giao dịch trung bình theo độ tuổi
plt.subplot(2, 3, 1)
age_trans_all = final_client_df.groupby('age_group')['total_transactions'].mean()
age_trans_credit = credit_card_users.groupby('age_group')['total_transactions'].mean()

plt.bar(x - width/2, age_trans_all.values, width, label='Tổng thể', alpha=0.7, color='lightgreen')
plt.bar(x + width/2, age_trans_credit.values, width, label='Credit Card', alpha=0.7, color='gold')
plt.title('Số giao dịch trung bình theo độ tuổi', fontsize=12, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Số giao dịch trung bình')
plt.xticks(x, age_trans_all.index)
plt.legend()
plt.grid(axis='y', alpha=0.3)


# Tổng nợ trung bình theo độ tuổi
plt.subplot(2, 3, 2)  # Vị trí mới
age_debt_all = final_client_df.groupby('age_group')['total_debt'].mean()
age_debt_credit = credit_card_users.groupby('age_group')['total_debt'].mean()

plt.bar(x - width/2, age_debt_all.values, width, label='Tổng thể', alpha=0.7, color='lightcoral')
plt.bar(x + width/2, age_debt_credit.values, width, label='Credit Card', alpha=0.7, color='mediumpurple')
plt.title('Tổng nợ trung bình theo độ tuổi', fontsize=12, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Tổng nợ trung bình ($)')
plt.xticks(x, age_debt_all.index)
plt.legend()
plt.grid(axis='y', alpha=0.3)
```

</details>

![i4](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/Card_age.png)
![i5](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/Credit_limit_age_income.png)
![i6](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/Transactions_debt_age.png)

*Insight:*
- Phân bố tỷ lệ các loại thẻ tương đối đồng đều giữa các nhóm tuổi, với tỷ lệ khách hàng cao trong tổng thể, hiện tại số lượng thẻ Credit đang được dùng cao nhất ở nhóm khách hàng 35-54 tuổi, điều này cũng đúng với tần suất chi tiêu ở nhóm tuổi này
- So với các nhóm khách hàng khác, mức tổng nợ trung bình của khách hàng ở nhóm tuổi 35-54 ở mức trung bình, không có nhiều sự khác biệt với tổng thể, minh chứng hiện thực nhu cầu vay của nhóm khách hàng dùng thẻ Credit không hề cao trong cả thập kỷ
- Phân bố trên heatmap cho thấy thu nhập chủ yếu của nhóm khách dùng thẻ Credit hàng đang ở mức trung bình từ 25k $ - 75k $ tương đương với phân phối của tổng thể khách hàng. Tương tự, cũng không có nhiều sự khác biệt giữa hạn mức tín dụng nhóm khách hàng sử dụng thẻ Credit với tổng thể, cả hai nhóm đều có mức hạn mức từ thấp đến trung bình. Điều này có thể lý giải bởi thói quen chi tiêu, với thẻ Debit là chi tiêu có phần thoáng hơn nhưng chi tiêu dựa vào số tiền sẵn có trong tài khoản khách hàng từ đó nhu cầu vay và sử dụng tín dụng không cao, trong khi những khách hàng sủ dụng Credit lại có thói quen chi tiêu tín dụng hạn chế.

# So sánh và đánh giá sự khác biệt về rủi ro giữa nhóm khách hàng sử dụng Credit với tổng các nhóm khách hàng bằng các biến điểm tín dụng và DTI

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
```
print('Điểm tín dụng ở mọi mức điểm đều ở tỷ lệ cân bằng so với tổng thể, đánh giá ít rủi ro')

# Tạo nhóm điểm tín dụng
def categorize_credit_score(score):
    if score >= 750:
        return 'Cao (750+)'
    elif score >= 650:
        return 'Trung bình (650-749)'
    else:
        return 'Thấp (<650)'

# Áp dụng cho cả hai dataset
final_client_df['credit_score_group'] = final_client_df['credit_score'].apply(categorize_credit_score)
credit_card_users['credit_score_group'] = credit_card_users['credit_score'].apply(categorize_credit_score)
# === TÍNH TOÁN TỶ LỆ NỢ/THU NHẬP (DTI) ===

final_client_df['dti_ratio'] = final_client_df['total_debt'] / final_client_df['yearly_income']
credit_card_users['dti_ratio'] = credit_card_users['total_debt'] / credit_card_users['yearly_income']

# SO SÁNH PHÂN BỐ ĐIỂM TÍN DỤNG VÀ DTI
plt.figure(figsize=(20, 15))

# Định nghĩa màu sắc
credit_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
credit_labels = ['Thấp (<650)', 'Trung bình (650-749)', 'Cao (750+)']

dti_colors = ['#2E8B57', '#3CB371', '#FFD700', '#FF6347', '#D3D3D3']
dti_labels = ['Rất thấp (0-0.2)', 'Thấp (0.2-0.36)', 'Trung bình (0.36-0.5)', 'Cao (>0.5)', 'Không xác định']

# Phân bố điểm tín dụng theo độ tuổi - Tổng thể
plt.subplot(2, 3, 1)
credit_age_all = pd.crosstab(final_client_df['age_group'], final_client_df['credit_score_group'])
credit_age_all_percent = credit_age_all.div(credit_age_all.sum(axis=1), axis=0) * 100
credit_age_all_percent.plot(kind='bar', stacked=True, ax=plt.gca(), color=credit_colors, legend=False)
plt.title('PHÂN BỐ ĐIỂM TÍN DỤNG THEO ĐỘ TUỔI\n(Tổng thể - %)', fontsize=12, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Tỷ lệ phần trăm (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Phân bố điểm tín dụng theo độ tuổi - Credit Card
plt.subplot(2, 3, 2)
credit_age_credit = pd.crosstab(credit_card_users['age_group'], credit_card_users['credit_score_group'])
credit_age_credit_percent = credit_age_credit.div(credit_age_credit.sum(axis=1), axis=0) * 100
credit_age_credit_percent.plot(kind='bar', stacked=True, ax=plt.gca(), color=credit_colors, legend=False)
plt.title('Phân bố điểm tín dụng theo độ tuổi\n(Khách hàng dùng thẻ tín dụng - %)', fontsize=12, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Tỷ lệ phần trăm (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.figlegend(handles=[plt.Rectangle((0,0),1,1, fc=color) for color in colors],
            labels=legend_labels,
            title='Nhóm điểm tín dụng',
            loc='upper center',
            bbox_to_anchor=(0.5, 0.95),
            ncol=3,
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True)

# DTI theo độ tuổi
plt.subplot(2, 3, 3)
age_dti_all = final_client_df.groupby('age_group')['dti_ratio'].mean()
age_dti_credit = credit_card_users.groupby('age_group')['dti_ratio'].mean()

x = np.arange(len(age_dti_all.index))
width = 0.35

plt.bar(x - width/2, age_dti_all.values, width, label='Tổng thể', alpha=0.7, color='lightblue')
plt.bar(x + width/2, age_dti_credit.values, width, label='Credit Card', alpha=0.7, color='lightcoral')
plt.title('Tỷ lệ Nợ/Thu nhập (DTI) trung bình\nTheo độ tuổi', fontsize=12, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('DTI Ratio')
plt.xticks(x, age_dti_all.index, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Phân nhóm DTI
def categorize_dti(dti):
    if dti <= 0.2:
        return 'Rất thấp (0-0.2)'
    elif dti <= 0.3:
        return 'Thấp (0.2-0.3)'
    elif dti <= 0.4:
        return 'Trung bình (0.3-0.4)'
    else:
        return 'Cao (>0.4)'

final_client_df['dti_group'] = final_client_df['dti_ratio'].apply(categorize_dti)
credit_card_users['dti_group'] = credit_card_users['dti_ratio'].apply(categorize_dti)

dti_all = final_client_df['dti_group'].value_counts(normalize=True) * 100
dti_credit = credit_card_users['dti_group'].value_counts(normalize=True) * 100

# Pie chart cho tổng thể
plt.subplot(2, 3, 4)
plt.pie(dti_all.values, labels=dti_all.index, autopct='%1.1f%%', startangle=90, 
        colors=dti_colors[:len(dti_all)])
plt.title('Phân bố DTI cho tổng thể', fontsize=12, fontweight='bold')

# Pie chart cho credit card
plt.subplot(2, 3, 5)
plt.pie(dti_credit.values, labels=dti_credit.index, autopct='%1.1f%%', startangle=90,
        colors=dti_colors[:len(dti_credit)])
plt.title('Phân bố DTI cho khách hàng dùng thẻ tín dụng', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.92])  
plt.show()
```
</details>

# Phân bố nhóm DTI cao, diễn giải lý do DTI cao

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
```
# Lọc nhóm DTI cao
high_dti_clients = final_client_df[final_client_df['dti_ratio'] > 0.4]

# Phân nhóm thu nhập cho DTI cao

high_dti_clients['income_group'] = final_client_df['group_income']
income_dist_high_dti = high_dti_clients['income_group'].value_counts(normalize=True) * 100

# Phân tích nợ của nhóm DTI cao
print("\n2. PHÂN TÍCH THEO TỔNG NỢ:")
debt_stats_high_dti = high_dti_clients['total_debt'].describe()
debt_stats_all = final_client_df['total_debt'].describe()

print("Thống kê tổng nợ - Nhóm DTI cao vs Tổng thể:")
print(f"  - Nợ trung bình: ${debt_stats_high_dti['mean']:,.0f} (DTI cao) vs ${debt_stats_all['mean']:,.0f} (Tổng thể)")
print(f"  - Nợ trung vị: ${debt_stats_high_dti['50%']:,.0f} (DTI cao) vs ${debt_stats_all['50%']:,.0f} (Tổng thể)")

# Phân nhóm nợ
debt_bins = [0, 10000, 25000, 50000, 100000, float('inf')]
debt_labels = ['<10k', '10k-25k', '25k-50k', '50k-100k', '>100k']

high_dti_clients['debt_group'] = pd.cut(high_dti_clients['total_debt'], bins=debt_bins, labels=debt_labels)
debt_dist_high_dti = high_dti_clients['debt_group'].value_counts(normalize=True) * 100

# TẠO BIỂU ĐỒ CHỨNG MINH
plt.figure(figsize=(18, 12))

# Mối quan hệ Thu nhập vs DTI
plt.subplot(2, 3, 1)
plt.scatter(final_client_df['yearly_income'], final_client_df['dti_ratio'], 
           alpha=0.6, color='blue', label='Tổng thể', s=20)
plt.scatter(high_dti_clients['yearly_income'], high_dti_clients['dti_ratio'], 
           alpha=0.8, color='red', label='DTI > 0.4', s=30)
plt.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='Ngưỡng DTI cao (0.4)')
plt.title('Mối quan hệ Thu nhập vs DTI', fontsize=12, fontweight='bold')
plt.xlabel('Thu nhập hàng năm ($)')
plt.ylabel('Tỷ lệ DTI')
plt.legend()
plt.grid(True, alpha=0.3)

# Heatmap Thu nhập vs Tổng nợ cho DTI cao
plt.subplot(2, 3, 2)
heatmap_data = pd.crosstab(high_dti_clients['income_group'], 
                          high_dti_clients['debt_group'], 
                          normalize='index') * 100
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='Reds', cbar_kws={'label': 'Tỷ lệ %'})
plt.title('Phân bố DTI cao: Thu nhập vs Tổng nợ', fontsize=12, fontweight='bold')
plt.xlabel('Nhóm tổng nợ')
plt.ylabel('Nhóm thu nhập')

plt.tight_layout()
plt.show()

# Tính tỷ lệ khách hàng có thu nhập thấp (<50k) trong nhóm DTI cao
low_income_high_dti = len(high_dti_clients[high_dti_clients['yearly_income'] < 50000])
low_income_rate = low_income_high_dti / len(high_dti_clients) * 100

# Tính tỷ lệ khách hàng có nợ cao (>25k) trong nhóm DTI cao
high_debt_high_dti = len(high_dti_clients[high_dti_clients['total_debt'] > 25000])
high_debt_rate = high_debt_high_dti / len(high_dti_clients) * 100
```
</details>

![i7](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/Credit_score_DTI.png)
![i8](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/high_DTI_income.png)
*Insight:*
- Điểm tín dụng ở mọi mức điểm đều ở tỷ lệ cân bằng so với tổng thể, đánh giá ít rủi ro. Tuy nhiên tỷ lệ DTI lại ở mức cao đối với cả khách hàng sử dụng Credit và tổng thể
- Có thể thấy nhóm khách hàng có tỷ lệ DTI cao chủ yếu có mức thu nhập dưới 50k $ và có tổng nợ hơn 50k $ (tức là nợ nhiều trong khi thu nhập không nhiều hơn tương ứng hoặc nợ ít tuy nhiên thu nhập cũng hạn chế tương ứng). Điều này có thể lý giải bởi tệp khách hàng tổng thể của tổ chức tài chính này chủ yếu là người có thu nhập trung bình, trong khi nợ trung bình khách hàng tập trung ở mức tương đương.

* Áp dụng ưu đãi vay đối với dịch vụ nào?

<details>
<summary><b>Xem toàn bộ code</b></summary>
    
```
print(f"Số lượng khách hàng dùng credit card: {len(credit_card_users):,}")
print(f"Tỷ lệ: {len(credit_card_users)/len(final_client_df)*100:.1f}% tổng số khách hàng")
print('Tỷ lệ sử dụng Credit card để chi tiêu cho du lịch cao hơn so với tổng thể gần 4%') 

# SO SÁNH PHÂN BỐ DANH MỤC CHI TIÊU - TỔNG THỂ vs CREDIT CARD
plt.figure(figsize=(20, 10))

# Tính toán phần trăm cho cả hai nhóm
category_counts_all = final_client_df['top_category'].value_counts()
category_percent_all = (category_counts_all / len(final_client_df) * 100).round(1)

category_counts_credit = credit_card_users['top_category'].value_counts()
category_percent_credit = (category_counts_credit / len(credit_card_users) * 100).round(1)

# Tạo màu sắc thống nhất cho cả hai pie chart
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

# Pie Chart 1: Tổng thể
plt.subplot(1, 2, 1)
wedges1, texts1, autotexts1 = plt.pie(category_percent_all.values, 
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      textprops={'fontsize': 10, 'fontweight': 'bold'})

# Làm đẹp phần trăm bên trong
for autotext in autotexts1:
    autotext.set_color('white')

# Tạo chú thích
legend_labels1 = [f'{category} ({count:,} KH)' for category, count in category_counts_all.items()]
plt.legend(wedges1, legend_labels1, 
           title="Danh mục chi tiêu",
           title_fontsize=12,
           loc="upper right",
           bbox_to_anchor=(1.2, 1.0),
           fontsize=9)

plt.title('Phân bố danh mục chi tiêu\nTổng thể', fontsize=16, fontweight='bold')
plt.axis('equal')

# Pie Chart 2: Credit Card Users
plt.subplot(1, 2, 2)
wedges2, texts2, autotexts2 = plt.pie(category_percent_credit.values, 
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      textprops={'fontsize': 10, 'fontweight': 'bold'})

# Làm đẹp phần trăm bên trong
for autotext in autotexts2:
    autotext.set_color('white')

# Tạo chú thích
legend_labels2 = [f'{category} ({count:,} KH)' for category, count in category_counts_credit.items()]
plt.legend(wedges2, legend_labels2, 
           title="Danh mục chi tiêu",
           title_fontsize=12,
           loc="upper right",
           bbox_to_anchor=(1.2, 1.0),
           fontsize=9)

plt.title('Phân bố danh mục chi tiêu\nKhách hàng dùng thẻ tín dụng', fontsize=16, fontweight='bold')
plt.axis('equal')

plt.tight_layout()
plt.show()
```
</details>

<details>
<summary><b>Xem toàn bộ code</b></summary>
    
```
print('Có thể thấy, nhóm tuổi 35-54 có mức chi tiêu cho Du lịch và Giải trí tương đương với các nhóm tuổi khác, tuy nhiên, chi tiêu cho Ăn uống cũng như Xăng dầu & Di chuyển lại ở mức cao. Có khả năng chi tiêu, có thể áp dụng gói dịch vụ cho vay đi Du lịch, giúp kích thích tăng trưởng chi tiêu cho sản phẩm liên quan đến Du lịch & Giải trí')

# Tìm top 3 category lớn nhất trong credit card users
top3_categories = credit_card_users['top_category'].value_counts().head(3).index.tolist()
print(f"Top 3 mặt hàng chi tiêu nhiều nhất của khách hàng sử dụng thẻ tín dụng:")
for i, category in enumerate(top3_categories, 1):
    count = credit_card_users[credit_card_users['top_category'] == category].shape[0]
    percent = (count / len(credit_card_users)) * 100
    print(f"{i}. {category}: {count:,} khách hàng ({percent:.1f}%)")

print("\n")

# Lọc chỉ top 3 category
top3_data = credit_card_users[credit_card_users['top_category'].isin(top3_categories)]

# Tính chi tiêu trung bình theo độ tuổi và category
age_category_spent = top3_data.groupby(['age_group', 'top_category'])['total_spent'].mean().unstack()

# Điền giá trị 0 cho các ô NaN
age_category_spent = age_category_spent.fillna(0)

# Vẽ biểu đồ
plt.figure(figsize=(14, 8))

# Biểu đồ cột grouped
x = np.arange(len(age_category_spent.index))
width = 0.25

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, category in enumerate(top3_categories):
    plt.bar(x + i*width - width, age_category_spent[category], width, 
            label=category, color=colors[i], alpha=0.8)

plt.xlabel('Nhóm tuổi', fontsize=12)
plt.ylabel('Chi tiêu trung bình ($)', fontsize=12)
plt.title('Chi tiêu trung bình bằng thẻ tín dụng\nTheo độ tuổi và Top 3 mặt hàng giao dịch nhiều nhất', 
          fontsize=14, fontweight='bold')
plt.xticks(x, age_category_spent.index)
plt.legend(title='Danh mục', title_fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Thêm giá trị trên các cột
for i, age_group in enumerate(age_category_spent.index):
    for j, category in enumerate(top3_categories):
        value = age_category_spent.loc[age_group, category]
        plt.text(i + j*width - width, value + 50, f'${value:,.0f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
```
</details>

![i9](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/categories.png)
![i10](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/top3_categories.png)

*Insight:*
- Tỷ lệ sử dụng Credit card để chi tiêu cho du lịch cao hơn so với tổng thể gần 4%
- Có thể thấy, nhóm tuổi 35-54 có mức chi tiêu cho Du lịch và Giải trí tương đương với các nhóm tuổi khác, tuy nhiên, chi tiêu cho Ăn uống cũng như Xăng dầu & Di chuyển lại ở mức cao. Có khả năng chi tiêu, có thể áp dụng gói dịch vụ cho vay đi Du lịch, giúp kích thích tăng trưởng chi tiêu cho sản phẩm liên quan đến Du lịch & Giải trí

# Phân tích sâu hơn nhóm khách hàng 35-54
* Tạo bảng dữ liệu cho nhóm khách hàng 35-54, tạo chỉ số đánh giá như hạn mức/thu nhập. Ở phần này, phân tích sâu tình trạng hạn mức và đề xuất chính sách hạn mức dựa vào thu nhập của nhóm tuổi 35-54
    
```
client_analysis = final_client_df.copy()

# Tính tỷ lệ hạn mức tín dụng so với thu nhập (ước tính)
# Chuyển yearly_income về dạng số để tính toán
income_mapping = {
    '<25000': 12500,
    '25000–50000': 37500,
    '50000–75000': 62500,
    '75000–100000': 87500,
    '100000–150000': 125000,
    '150000+': 175000
}

client_analysis['income_numeric'] = client_analysis['group_income'].astype(str).map(income_mapping)
client_analysis['credit_limit_income_ratio'] = client_analysis['credit_limit'] / client_analysis['income_numeric']

# Tính tỷ lệ sử dụng credit_limit (ước tính từ chi tiêu)
client_analysis['credit_utilization_ratio'] = (client_analysis['total_spent'] / client_analysis['credit_limit']).clip(upper=1)  # Giới hạn max 100%

# Phân loại khách hàng theo độ tuổi mục tiêu (35-55)
client_analysis['is_target_age'] = client_analysis['age_group'].isin(['35-44', '45-54'])

print(f"\nSố khách hàng trong nhóm mục tiêu (35-54 tuổi): {client_analysis['is_target_age'].sum():,}")
print(f"Tỷ lệ: {client_analysis['is_target_age'].mean()*100:.1f}% tổng số khách hàng")
```

* Phân tích hạn mức và so sánh với các nhóm tuổi bằng chỉ số, từ đó thăm dò nhu cầu tăng tín dụng của nhóm khách hàng. Ở đây, so sánh tỷ lệ hạn mức/thu nhập với mức DTI an toàn được quy định ở châu Âu, từ đó đánh giá dư nợ tăng hạn mức của khách hàng

<details>
<summary><b>Xem toàn bộ code</b></summary>
    
```
# Hạn mức theo độ tuổi và thu nhập
plt.figure(figsize=(20, 12))

# Tỷ lệ Han mức/Thu nhập theo độ tuổi
plt.subplot(2, 3, 1)
ratio_by_age = client_analysis.groupby('age_group')['credit_limit_income_ratio'].mean()
plt.bar(ratio_by_age.index, ratio_by_age.values, color='lightcoral', alpha=0.7)
plt.title('Tỷ lệ hạn mức/Thu nhập theo độ tuổi', fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Tỷ lệ Hạn mức/Thu nhập')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Biểu đồ 2: Tỷ lệ sử dụng credit limit theo độ tuổi
plt.subplot(2, 3, 2)
utilization_by_age = client_analysis.groupby('age_group')['credit_utilization_ratio'].mean()
plt.bar(utilization_by_age.index, utilization_by_age.values * 100, color='lightgreen', alpha=0.7)
plt.title('Tỷ lệ sử dụng hạn mức theo độ tuổi', fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Tỷ lệ sử dụng (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Biểu đồ 3: Phân bố credit limit cho nhóm mục tiêu vs các nhóm khác
plt.subplot(2, 3, 4)
target_group = client_analysis[client_analysis['is_target_age']]['credit_limit']
other_group = client_analysis[~client_analysis['is_target_age']]['credit_limit']

plt.hist(target_group, bins=30, alpha=0.5, label='Nhóm 35-54 tuổi', color='red', density=True)
plt.hist(other_group, bins=30, alpha=0.5, label='Các nhóm khác', color='blue', density=True)
plt.title('Phân bố hạn mức tín dụng\nNhóm mục tiêu vs Các nhóm tuổi khác', fontweight='bold')
plt.xlabel('Hạn mức tín dụng ($)')
plt.ylabel('Tần suất')
plt.legend()
plt.grid(alpha=0.3)

# Biểu đồ 4: Tỷ lệ khách hàng có credit_limit thấp so với thu nhập
def identify_need_credit_increase(row):
    """Xác định khách hàng cần tăng credit limit"""
    if row['credit_limit_income_ratio'] < 0.15:  # Credit limit < 10% thu nhập
        return 'Cần tăng mạnh'
    elif row['credit_limit_income_ratio'] < 0.25:  # Credit limit < 20% thu nhập
        return 'Cần tăng vừa'
    elif row['credit_utilization_ratio'] > 0.8:  # Sử dụng > 80% credit limit
        return 'Đang sử dụng cao'
    else:
        return 'Ổn định'

client_analysis['credit_need'] = client_analysis.apply(identify_need_credit_increase, axis=1)

plt.subplot(2, 3, 5)
credit_need_by_age = pd.crosstab(client_analysis['age_group'], client_analysis['credit_need'], normalize='index') * 100
credit_need_by_age.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Nhu cầu tăng hạn mức theo độ tuổi', fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Tỷ lệ (%)')
plt.legend(title='Nhu cầu', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```
</details>

<details>
<summary><b>Xem toàn bộ code</b></summary>
    
```
target_customers = client_analysis[client_analysis['is_target_age']].copy()

# Thống kê cơ bản
print(f"Tỷ lệ Hạn mức tín dụng/Thu nhập trung bình: {target_customers['credit_limit_income_ratio'].mean():.2f}")
print(f"Tỷ lệ sử dụng hạn mức trung bình: {target_customers['credit_utilization_ratio'].mean()*100:.1f}%")

# Phân tích theo thu nhập
plt.figure(figsize=(16, 10))

#  Scatter plot hạn mức vs Thu nhập 
plt.subplot(2, 3, 1)
plt.scatter(target_customers['income_numeric'], target_customers['credit_limit'], 
           alpha=0.6, c=target_customers['credit_utilization_ratio'], cmap='viridis')
plt.colorbar(label='Tỷ lệ sử dụng')
plt.xlabel('Thu nhập ước tính ($)')
plt.ylabel('Hạn mức tín dụng ($)')
plt.title('Hạn mức tín dụng vs Thu Nhập\n(Nhóm 35-54 tuổi)', fontweight='bold')
plt.grid(alpha=0.3)

# Vẽ đường chuẩn (credit limit = 40% thu nhập được coi là mức giới hạn an toàn ở châu Âu)
x_range = np.linspace(target_customers['income_numeric'].min(), target_customers['income_numeric'].max(), 100)
plt.plot(x_range, x_range * 0.40, 'r--', linewidth=2, label='Mức giới hạn an toàn: 40% thu nhập')
plt.legend()

# Phân bố tỷ lệ hạn mức/thu nhập
plt.subplot(2, 3, 2)
plt.hist(target_customers['credit_limit_income_ratio'], bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=0.4, color='red', linestyle='--', linewidth=2, label='Mức giới hạn toàn: 40%')
plt.xlabel('Tỷ lệ Hạn mức tín dụng/Thu nhập')
plt.ylabel('Số lượng khách hàng')
plt.title('Phân bố tỷ lệ Hạn mức tín dụng/Thu nhập\n(Nhóm 35-54 tuổi)', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# Mối quan hệ giữa điểm tín dụng và hạn mức
plt.subplot(2, 3, 4)
plt.scatter(target_customers['credit_score'], target_customers['credit_limit'], alpha=0.6)
plt.xlabel('Điểm tín dụng')
plt.ylabel('Hạn mức tín dụng ($)')
plt.title('Điểm tín dụng vs Hạn mức tín dụng\n(Nhóm 35-54 tuổi)', fontweight='bold')
plt.grid(alpha=0.3)

# Đề xuất chính sách theo nhóm thu nhập
plt.subplot(2, 3, 5)
policy_by_income = pd.crosstab(target_customers['group_income'], target_customers['credit_need'])
policy_by_income.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Đề xuất chính sách theo thu nhập', fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Số lượng khách hàng (35-54 tuổi)')
plt.legend(title='Nhu cầu', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```
</details>

![i11](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/target_age_group1.png)
![i12](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/target_age_group2.png)

*Insight:*
- Các dẫn chứng cho thấy nhóm tuổi 35-54 đang là nhóm tuổi có hạn mức thấp nhất so với thu nhập tiềm năng của họ, tuy nhiên vậy, tỷ lệ sử dụng hạn mức đang ở mức khá tốt, cao nhất trong mọi nhóm tuổi. Nhìn vào nhu cầu tăng trưởng hạn mức, nhóm tuổi 35-54 đang là 1 trong 2 nhóm có tỷ lệ cần tăng mạnh lớn nhất ở các nhóm tuổi
- Tỷ lệ Hạn mức tín dụng/Thu nhập trung bình: 0.32
- Tỷ lệ sử dụng hạn mức trung bình: 42.8%
- Nhóm tuổi 35-54 có tình trạng để trống hạn mức khi tỷ lệ sử dụng hạn mức là tương đối thấp ở mức hạn mức cao, hầu hết những người có tỷ lệ sử dụng tín dụng cao đều có mức giới hạn tín dụng thấp. Trong khi đó hạn mức tín dụng/thu nhập chủ yếu tập trung ở mức 0.2 - 0.3 (an toàn cao). Điểm tín dụng chủ yếu >650 nhưng lại ở hạn mức thấp. Các yếu tố đó chỉ ra rằng nhóm tuổi 35-54 có tiềm năng tăng trưởng tỷ lệ sử dụng tín dụng, phù hợp áp dụng các ưu đãi và chính sách tăng hạn mức ở nhóm đối tượng này


















