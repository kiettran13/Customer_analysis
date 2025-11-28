# Phân tích hành vi khách hàng
# Mục tiêu dự án
Mục tiêu của dự án là phân loại và xác định nhóm khách hàng phù hợp để thiết kế, đề xuất các sản phẩm – dịch vụ tín dụng tương ứng, giúp hoàn thiện chất lượng dịch vụ tín dụng và khuyến khích nhu cầu vay vốn tại tổ chức tài chính mục tiêu.
# Tổng quan dự án 
Dự án tập trung khai thác và phân tích hành vi giao dịch của khách hàng thông qua các đặc điểm nhân khẩu học kết hợp với mô hình chi tiêu thực tế. Đồng thời, phân tích xem xét các yếu tố ảnh hưởng đến việc sử dụng thẻ tín dụng (Credit Card) như độ tuổi, mức thu nhập, điểm tín dụng,… Qua đó, dự án hướng đến việc đánh giá sức khỏe tài chính, tình trạng tín dụng và thói quen chi tiêu của từng nhóm khách hàng, từ đó phân nhóm và đề xuất các sản phẩm tín dụng phù hợp với từng phân khúc.
Bộ dữ liệu sử dụng được trích xuất từ [Kaggle](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets). Đây là tập dữ liệu tài chính quy mô lớn của một tổ chức tài chính tại Mỹ gồm 5 file chính:
- Transaction Data (transactions_data.csv)
- Card Information (cards_data.csv)
- User Data (users_data.csv)
- Merchant Category Codes (mcc_codes.json)
- Fraud Labels (train_fraud_labels.json)

Bộ dữ liệu được chia sẻ ở trong folder [data](https://github.com/kiettran13/Customer_analysis/tree/964c99f7904bc7fa2fb1dace7d3f2466fff7a94f/data). Riêng dữ liệu giao dịch (Transaction Data) và dữ liệu về dấu hiệu lừa đảo (Fraud Labels) có kích thước lớn nên sẽ được chia sẻ thông qua đường link riêng. 
Mô tả chi tiết các biến trong bộ dữ liệu được tổng hợp trong file [variables](https://github.com/kiettran13/Customer_analysis/blob/main/variable)

# Chi tiết dự án:
# Tiền xử lý dữ liệu:
* Nhập dữ liệu và thư viện cần thiết:

<details>
<summary><b>Xem toàn bộ code </b></summary>

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
<details>
<summary><b>Xem toàn bộ code </b></summary>
  
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
</details>

* Kiểm tra NA và trùng lặp
<details>
<summary><b>Xem toàn bộ code </b></summary>
  
```
print(transactions_data.isna().sum())
print(users_data.isna().sum())
print(cards_data.isna().sum())
print(mcc_codes.isna().sum())
print(frauds_df.isna().sum())

print(transactions_data.duplicated().sum())
print(users_data.duplicated().sum())
print(cards_data.duplicated().sum())
print(mcc_codes.duplicated().sum())
print(frauds_df.duplicated().sum())
```
</details>

*Nhận xét:* Tất cả các trường dữ liệu được sử dụng đều không có NA và hiện tượng trùng lặp
* Xoá bỏ các giao dịch gian lận thông qua việc đối chiếu mã giao dịch với mã giao dịch được đánh dấu là lửa đảo, đảm bảo dữ liệu có ý nghĩa thống kê

<details>
<summary><b>Xem toàn bộ code </b></summary>
  
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
</details>

* Chuẩn hoá kiểu dữ liệu cho các biến mang giá trị ngày tháng và tiền tệ (giá trị giao dịch, thu nhập hàng năm, tổng nợ, hạn mức tín dụng) thành dạng số

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
    
    # 5. Xăng dầu & Phương tiện
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
* Thêm biến 'year' giảm số quan sát trong bảng transactions, 'spend' lọc ra các giao dịch trừ tiền, có ý nghĩa với việc phân tích thói quen tiêu dùng.
<details>
<summary><b>Xem toàn bộ code </b></summary>
  
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
</details>

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
<summary><b>Xem toàn bộ code </b></summary>
    
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

<details>
<summary><b>Xem toàn bộ code </b></summary>
  
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
</details>

* Chọn từ bảng ra các cột cần phân tích, lưu vào một bảng mới (final_client_df)
<details>
<summary><b>Xem toàn bộ code </b></summary>
  
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

# Phân nhóm nợ    
debt_bins = [0, 5000, 10000, 30000, 60000, 100000, np.inf]
debt_labels = [
    '<5000', 
    '5000-10000',
    '10000–30000', 
    '30000–60000', 
    '60000–100000',  
    '100000+'
]
client_df['group_debt'] = pd.cut(client_df['total_debt'], bins=debt_bins, labels=debt_labels, right=False)

# Chọn các cột quan trọng cho phân tích
final_client_df = client_df[[
    'client_id', 'gender', 'age_group', 'yearly_income', 'credit_score', 'credit_limit', 'group_income', 'total_debt',
    'card_type', 'total_spent', 'total_transactions', 'year', 'top_category', 'group_spent'
]]
print(final_client_df.head())
```
</details>

* Ở phần này, tiếp tục gộp các cột có số lượng biến lớn như tuổi, thu nhập, chi tiêu, làm gọn bộ dữ liệu nhưng vẫn giữ các cột gốc trong bảng client_df, chỉ chọn lấy những biến đã được gộp theo nhóm vào final_client_df để thuận tiện cho EDA.  Các biến sau đó nếu được tạo thêm, hoặc các bảng dữ liệu con sẽ được tạo từ bảng này.
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
* Ở bước này, tôi sẽ trực quan hoá dữ liệu để khám phá tổng quan tính chất các biến và một số quan hệ cơ bản của biến dựa vào 3 bảng đã được xử lý: 'final_client_df', 'client_static_info', 'client_dynamic_info'. Sau đó, tiếp tục phân tích sâu hơn nhóm khách hàng sử dụng thẻ Credit nhằm tìm kiếm insight phù hợp với mục đích dự án
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
*Nhận xét:*
- Phân bố nhân khẩu học của các khách hàng khá đồng đều giữa các nhóm, phân bố độ tuổi giao động trong khoảng 5% giữa các nhóm, khách hàng 45-54 chiếm tỷ lệ lớn nhất (25,3%), khách hàng nhóm tuổi dưới 25 chiếm tỷ lệ nhỏ nhất, chỉ 0,3%
- Phân bố thu nhập hàng năm phổ biến nhất ở mức 25k-50k $ và 50k-75k $, đây là mức trung bình trong phổ dữ liệu, khách hàng của tổ chức ngân hàng chủ yếu là phân lớp bình dân đến trung lưu.
- Phân bố loại thẻ và số giao dịch và mặt hàng chi tiêu cho thấy, có phần lớn khách hàng giao dịch thường xuyên xung mức trung bình 1000 lần giao dịch mỗi năm trong hầu hết thời gian, và loại thẻ họ chủ yếu sử dụng trong giao dịch là Debit và Credit với tỷ lệ khá cân đối, 55,5% cho Debit và 41,4% cho Credit và top 3 nhóm dịch vụ được khách hành sử dụng nhiều nhất là Thực phẩm & Ăn uống, Xăng dầu & Di chuyển, Du lịch $ Giải trí.
- Khách hàng có sự cẩn trọng trong xu hướng sử dụng nợ và chi tiêu nợ khi hạn mức tín dụng luôn ở mức thấp, trung bình dưới 20k $ cho mỗi năm và điểm tín dụng luôn cao hơn mức trung bình (>650) và một tổng nợ trung bình 1 khách hàng đang gánh chịu chỉ tập trung dưới 10k $ mỗi năm trong thời gian một thập kỷ.

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
yearly_group_income = pd.crosstab(final_client_df['year'], final_client_df['group_income'])
sns.heatmap(yearly_group_income, annot=True, cmap='YlGn', fmt='d')
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
*Nhận xét:*
- Hầu hết trong bộ dữ liệu đều có thay đổi qua các năm, tuy nhiên các thay đổi này hoàn toàn không đáng kể. Điều này cho thấy mức độ ổn định từ tệp khách hàng của tổ chức tài chính

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
*Nhận xét:*
- Ở khía cạnh thu nhập trung bình, những khách hàng dùng thẻ Credit nhìn chung có thu nhập đều và cao hơn nhóm sử dụng thẻ Debit ở mọi nhóm tuổi.
- Tuy nhiên với thu nhập gần tương đương nhau như vậy nhưng khi nhìn vào chi tiêu trung bình có thể thấy những khách hàng sử dụng thẻ Debit cho thấy sự phóng khoáng trong chi tiêu hơn, còn những khách hàng sử dụng thẻ Credit lại chi tiêu tương đối đồng đều giữa các nhóm tuổi và thấp hơn nhóm sử dụng thẻ Debit. Đây có thể là một dấu hiệu cho thấy chi tiêu dè dặt của nhóm dùng thẻ Credit bị đè nặng bởi một yếu tố khác ngoài thu nhập 

## Phân tích sâu hơn hành vi nhóm khách hàng sử dụng thẻ tín dụng 
* Ở phần này, tôi lọc những khách hàng sử dụng thẻ tín dụng (Credit) và sử dụng heatmap và subplot phân tích so sánh các biến hạn mức, tần suất giao dịch và tổng nợ giữa nhóm sử dụng thẻ tín dụng và tổng thể khách hàng, từ đó xem xét tính chất, thói quen nhóm khách hàng sử dụng thẻ tín dụng
## Phân bố loại thẻ theo nhóm tuổi & phân bố chi tiêu & thu nhập theo nhóm tuổi và loại thẻ

<details>
<summary><b>Xem toàn bộ code </b></summary>

```
final_client_df = final_client_df.drop_duplicates(subset='client_id', keep='first')
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

<details>
<summary><b>Xem toàn bộ code </b></summary>

```
## Tạo dataset cho credit card users và debit card users
credit_card_users = final_client_df[final_client_df['card_type'] == 'Credit']
debit_card_users = final_client_df[final_client_df['card_type'] == 'Debit']
debit_card_users = debit_card_users.drop_duplicates(subset='client_id', keep='first')
credit_card_users = credit_card_users.drop_duplicates(subset='client_id', keep='first')

# Tạo heatmap
plt.figure(figsize=(20, 15))
# 1. SO SÁNH PHÂN BỐ ĐỘ TUỔI VS THU NHẬP, XEM SỰ KHÁC NHAU CỦA NHÂN KHẨU HỌC

# Tổng thể
plt.subplot(2, 3, 1)
age_income_all = pd.crosstab(final_client_df['age_group'], final_client_df['group_income'])
sns.heatmap(age_income_all, annot=True, cmap='Blues', fmt='d')
plt.title('Phân bố độ tuổi và thu nhập\n(Tổng thể)', fontsize=14, fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Độ tuổi')

# Chỉ credit card users
plt.subplot(2, 3, 2)
age_income_credit = pd.crosstab(credit_card_users['age_group'], credit_card_users['group_income'])
sns.heatmap(age_income_credit, annot=True, cmap='Blues', fmt='d')
plt.title('Phân bố độ tuổi và thu nhập\n(Chỉ khách hàng sử dụng thẻ tín dụng)', fontsize=14, fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Độ tuổi')

plt.subplot(2, 3, 3)
age_income_debit = pd.crosstab(debit_card_users['age_group'], debit_card_users['group_income'])
sns.heatmap(age_income_debit, annot=True, cmap='Blues', fmt='d')
plt.title('Phân bố độ tuổi và thu nhập\n(Chỉ khách hàng sử dụng thẻ ghi nợ)', fontsize=14, fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Độ tuổi')


# SO SÁNH ĐIỂM TÍN DỤNG THEO ĐỘ TUỔI VÀ THU NHẬP

# Tổng thể
plt.subplot(2, 3, 4)
age_income_credit_limit_all = final_client_df.groupby(['age_group', 'group_income'])['credit_limit'].mean().unstack()
sns.heatmap(age_income_credit_limit_all, annot=True, cmap='RdYlBu', fmt='.0f', center=700)
plt.title('Hạn mức tín dụng trung bình trong các năm\n(Tổng thể)', fontsize=14, fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Độ tuổi')

# Chỉ credit card users
plt.subplot(2, 3, 5)
age_income_credit_limit_only = credit_card_users.groupby(['age_group', 'group_income'])['credit_limit'].mean().unstack()
sns.heatmap(age_income_credit_limit_only, annot=True, cmap='RdYlBu', fmt='.0f', center=700)
plt.title('Hạn mức tín dụng trung bình trong các năm\n(Chỉ khách hàng sử dụng thẻ tín dụng )', fontsize=14, fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Độ tuổi')

plt.subplot(2, 3, 6)
age_income_debit_limit_only = debit_card_users.groupby(['age_group', 'group_income'])['credit_limit'].mean().unstack()
sns.heatmap(age_income_debit_limit_only, annot=True, cmap='RdYlBu', fmt='.0f', center=700)
plt.title('Hạn mức tín dụng trung bình trong các năm\n(Chỉ khách hàng sử dụng thẻ ghi nợ )', fontsize=14, fontweight='bold')
plt.xlabel('Thu nhập')
plt.ylabel('Độ tuổi')

plt.tight_layout()
plt.show()

```
```
# SO SÁNH SỐ GIAO DỊCH VÀ TỔNG NỢ TRUNG BÌNH THEO ĐỘ TUỔI
plt.figure(figsize=(20, 15))

# Số giao dịch trung bình theo độ tuổi
plt.subplot(2, 3, 1)
age_trans_all = final_client_df.groupby('age_group')['total_transactions'].mean()
age_trans_credit = credit_card_users.groupby('age_group')['total_transactions'].mean()
age_groups = final_client_df['age_group'].unique()
x = np.arange(len(age_groups))
width = 0.35  

plt.bar(x - width/2, age_trans_all.values, width, label='Tổng thể', alpha=0.7, color='lightgreen')
plt.bar(x + width/2, age_trans_credit.values, width, label='Credit Card', alpha=0.7, color='gold')
plt.title('Số giao dịch trung bình theo độ tuổi', fontsize=12, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Số giao dịch trung bình')
plt.xticks(x, age_trans_all.index)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Tổng nợ trung bình theo độ tuổi
plt.subplot(2, 3, 2)
age_debt_all = final_client_df.groupby('age_group')['total_debt'].mean()
age_debt_credit = credit_card_users.groupby('age_group')['total_debt'].mean()
age_debt_debit = debit_card_users.groupby('age_group')['total_debt'].mean()

plt.bar(x - width, age_debt_all.values, width, label='Tổng thể', alpha=0.7, color='lightcoral')
plt.bar(x, age_debt_credit.values, width, label='Credit Card', alpha=0.7, color='mediumpurple')
plt.bar(x + width, age_debt_debit.values, width, label='Debit Card', alpha=0.7, color='lightblue')
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
![i6](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/transaction_debt_age.png)

*Nhận xét:*
- Tỷ lệ khách hàng sử dụng thẻ tín dụng tương đối đồng đều giữa các nhóm tuổi, điều này cũng tỷ lệ thuận với phân bố số lượng khách hàng theo nhóm tuổi
- Phân bố trên heatmap cho thấy thu nhập chủ yếu của nhóm khách dùng thẻ Credit hàng đang ở mức trung bình từ 25k$ - 75k$ tương đương với phân phối của nhóm khách hàng sử dụng thẻ Debit và tổng thể khách hàng. Tương tự, chính sách hạn mức thể hiện ra cả hai nhóm đều có mức hạn mức từ thấp đến trung bình đối với mức thu nhập thấp và trung, trong khi đó, nhóm dùng thẻ Credit có hạn mức trung bình hầu hết nhóm tuổi đều thấp hơn nhóm sử dụng thẻ Debit, điều này có thể ám chỉ mức tín nhiệm tín dụng hiện tại của tổ chức tài chính này với nhóm dùng Credit đang thấp hơn nhóm sử dụng Debit
- Trong khi đó, tần suất giao dịch trong các năm của khách hàng giữa 2 nhóm không có nhiều sự khác biệt. Trong khi đó tổng nợ trung bình của nhóm khách hàng 2 loại thẻ cũng tương đối giống nhau. Từ đây có thể thấy nhóm khách hàng sử dụng thẻ 

## Tổng kết phân tích EDA
- Có thể thấy tổ chức tài chính này có một tệp khách hàng ổn định với hành vi khách hàng thay đổi tương đối ít qua các năm. Đặc trưng bởi các biến về nhân khẩu học và các biến về tài chính cá nhân và hành vi tài chính. Với một tệp khách hàng chắc chắn và thói quen lâu năm, khách hàng có sự trung thành và gắn kết cao với dịch vụ của tổ chức ngân hàng, tuy nhiên mục đích chính để sử dụng thanh toán cho các mặt hàng thiết yếu như Ăn uống, Xăng dầu & Phương tiện và một phần Du lịch & Giải trí. Thông qua EDA tổng quát có thể thấy khách hàng chủ yếu thuộc nhóm bình dân đến trung lưu, lượng chi tiêu của khách hàng qua các năm tập trung ở mức 3k$ - 10k$ trong khi thu nhập năm của họ phân bố chủ yếu ở 25k$-75k$.
- Trong đó, tệp khách hàng sử dụng thẻ tín dụng đang là tệp khách hàng đông đảo, chiếm đến gần 42% khách hàng, đây là nhóm khách hàng đáng lẽ ra sẽ đem lại doanh thu tín dụng cao cho tổ chức thông qua chi tiêu tín dụng, nhưng dữ liệu cho thấy nhóm khách hàng này có chi tiêu thấp hơn nhóm Credit mặc dù hai nhóm có thu nhập và nợ trung bình tương đương nhau.
### Đặt giả thuyết
Dựa trên hành vi chi tiêu thấp và thu nhập cao hơn trung bình, kết hợp với việc nhóm Credit có tổng nợ tương đương Debit nhưng chi tiêu thấp hơn đáng kể, một giả thuyết hợp lý là họ có xu hướng duy trì nợ tín dụng, hạn chế vay thêm và do đó hạn chế chi tiêu qua thẻ tín dụng để tránh tăng dư nợ, ảnh hưởng đến điểm tín dụng vốn đang thấp hơn trung bình của họ

# PHÂN TÍCH GIẢ THUYẾT:

## Phân tích tình hình điểm tín dụng và so sánh phân bổ tỷ lệ DTI theo từng mức độ giữa 2 nhóm khách hàng sử dụng Debit và Credit

- Ở phần này của phân tích, tôi sẽ chia điểm tín dụng của khách hàng vào các nhóm điểm tín dụng theo FICO - thang chấm điểm tín dụng phổ biến ở Mỹ và chia chỉ số DTI vào các nhóm theo CFPB & Fannie Mae - hai cơ quan được chính phủ bảo lãnh và kiểm soát các khoản vay thế chấp ở Mỹ. Từ đó xác định chính xác tỷ lệ khách hàng theo nhóm điểm tín dụng và DTI.

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
```
def categorize_credit_score(score):
    if score >= 740:
        return 'Cao (740+)'
    elif score >= 670:
        return 'Trung bình (670-740)'
    else:
        return 'Thấp (<670)'

# Áp dụng phân loại cho dataset
debit_card_users['credit_score_group'] = debit_card_users['credit_score'].apply(categorize_credit_score)
credit_card_users['credit_score_group'] = credit_card_users['credit_score'].apply(categorize_credit_score)

# Công thức tính DTI
debit_card_users['dti_ratio'] = debit_card_users['total_debt'] / debit_card_users['yearly_income']
credit_card_users['dti_ratio'] = credit_card_users['total_debt'] / credit_card_users['yearly_income']

# Đặt bảng màu
credit_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
credit_labels = ['Thấp (<670)', 'Trung bình (670-740)', 'Cao (740+)']

dti_colors = ['#2E8B57', '#3CB371', '#FFD700', '#FF6347', '#D3D3D3']
dti_labels = ['Rất thấp (0-0.2)', 'Thấp (0.2-0.3)', 'Trung bình (0.3-0.36)', 'Cao (>0.36)', 'Không xác định']

# SO SÁNH PHÂN BỐ ĐIỂM TÍN DỤNG VÀ DTI
plt.figure(figsize=(20, 18))

# Phân bố điểm tín dụng theo độ tuổi - Debit Card
plt.subplot(2, 3, 1)
credit_age_debit = pd.crosstab(debit_card_users['age_group'], debit_card_users['credit_score_group'])
credit_age_debit_percent = credit_age_debit.div(credit_age_debit.sum(axis=1), axis=0) * 100
credit_age_debit_percent.plot(kind='bar', stacked=True, ax=plt.gca(), color=credit_colors, legend=False)
plt.title('Phân bố điểm tín dụng theo độ tuổi\n(nhóm dùng Debit - %)', fontsize=12, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Tỷ lệ phần trăm (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Phân bố điểm tín dụng theo độ tuổi - Credit Card
plt.subplot(2, 3, 2)
credit_age_credit = pd.crosstab(credit_card_users['age_group'], credit_card_users['credit_score_group'])
credit_age_credit_percent = credit_age_credit.div(credit_age_credit.sum(axis=1), axis=0) * 100
credit_age_credit_percent.plot(kind='bar', stacked=True, ax=plt.gca(), color=credit_colors, legend=False)
plt.title('Phân bố điểm tín dụng theo độ tuổi\n(nhóm dùng Credit - %)', fontsize=12, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Tỷ lệ phần trăm (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.figlegend(handles=[plt.Rectangle((0,0),1,1, fc=color) for color in credit_colors],
            labels=credit_labels,
            title='Nhóm điểm tín dụng',
            loc='upper center',
            bbox_to_anchor=(0.5, 0.05),
            ncol=3,
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True)

# Phân nhóm DTI
def categorize_dti(dti):
    if dti <= 0.2:
        return 'Rất thấp (0-0.2)'
    elif dti <= 0.3:
        return 'Thấp (0.2-0.3)'
    elif dti <= 0.36:
        return 'Trung bình (0.3-0.36)'
    else:
        return 'Cao (>0.36)'

debit_card_users['dti_group'] = debit_card_users['dti_ratio'].apply(categorize_dti)
credit_card_users['dti_group'] = credit_card_users['dti_ratio'].apply(categorize_dti)

dti_debit = debit_card_users['dti_group'].value_counts(normalize=True) * 100
dti_credit = credit_card_users['dti_group'].value_counts(normalize=True) * 100

# Chỉnh thứ tự bảng màu
dti_order = ['Rất thấp (0-0.2)', 'Thấp (0.2-0.3)', 'Trung bình (0.3-0.36)', 'Cao (>0.36)']
dti_debit = dti_debit.reindex(dti_order)
dti_credit = dti_credit.reindex(dti_order)

# Pie chart cho Debit card 
plt.subplot(2, 3, 4)
plt.pie(dti_debit.values, labels=dti_debit.index, autopct='%1.1f%%', startangle=90,
        colors=dti_colors[:len(dti_debit)])
plt.title('Phân bố DTI cho khách hàng dùng thẻ Debit', fontsize=12, fontweight='bold')

# Pie chart cho credit card 
plt.subplot(2, 3, 5)
plt.pie(dti_credit.values, labels=dti_credit.index, autopct='%1.1f%%', startangle=90,
        colors=dti_colors[:len(dti_credit)])
plt.title('Phân bố DTI cho khách hàng dùng thẻ Credit', fontsize=12, fontweight='bold')

# Legend for DTI
plt.figlegend(handles=[plt.Rectangle((0,0),1,1, fc=color) for color in dti_colors[:4]],
            labels=dti_labels[:4],
            title='Nhóm DTI',
            loc='lower center',
            bbox_to_anchor=(0.5, 0.05),
            ncol=4,
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  
plt.show()
```
</details>

![i7](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/credit_score_dti.png)
![i6](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/stat_total_credit_score.png)

*Nhận xét*
- Khi phân tích sâu vào điểm tín dụng và tỷ lệ nợ trên thu nhập (DTI) của hai nhóm khách hàng sử dụng Debit và Credit Card, các quan sát thu được củng cố rõ ràng cho giả thuyết đã đưa ra trong phần EDA. Nhóm khách hàng sử dụng Credit Card cho thấy điểm tín dụng trung bình thấp hơn ở các nhóm tuổi lớn, đồng thời có tới hơn 79.1% số khách hàng nằm trong mức DTI cao, trong khi chỉ 18,1% thuộc nhóm DTI rất thấp. Điều này phản ánh rằng phần lớn người dùng Credit Card đang duy trì mức dư nợ tín dụng lớn so với thu nhập – Đây là dấu hiệu rủi ro rõ ràng cần được xem xét khi cải thiện sản phẩm tín dụng và xem xét chính sách cho vay ở nhóm khách hàng này.
- Trong bối cảnh đó, việc họ hạn chế chi tiêu qua thẻ tín dụng trở nên hợp lý: khi điểm tín dụng không quá cao và tỷ lệ nợ đã vượt ngưỡng an toàn, khách hàng có xu hướng kiểm soát chi tiêu để tránh gia tăng dư nợ, bảo vệ điểm tín dụng và duy trì khả năng vay. Điều này phù hợp với thực tế quan sát được rằng mức chi tiêu trung bình hàng năm của nhóm Credit tương đối ổn định và thấp hơn nhóm sử dụng thẻ Debit, dù hai nhóm có thu nhập và tổng nợ trung bình tương đương.
- Mặt khác, nhóm khách hàng sử dụng Debit lại có tỷ lệ điểm tín dụng thấp thấp hơn so với nhóm Credit, trong khi tỷ lệ DTI cao của họ lại vượt cả nhóm Credit. Điều này cho thấy khách hàng Debit ít bị ảnh hưởng bởi áp lực duy trì điểm tín dụng hoặc hạn mức vay, từ đó có xu hướng chi tiêu thoải mái hơn. Đây là bằng chứng mạnh mẽ củng cố cho giả thuyết ở phần EDA: việc chi tiêu thấp của nhóm sử dụng Credit chủ yếu xuất phát từ áp lực dư nợ tín dụng và động cơ bảo vệ điểm tín dụng, chứ không đến từ năng lực tài chính hay khả năng chi trả.

# Phân tích rõ hơn cấu trúc nhóm khách hàng đang dùng thẻ Credit theo 2 nhóm DTI thấp và DTI cao nhằm tìm hướng phát triển sản phẩm riêng biệt:

* Ở phần này, tôi sẽ phân tích tình hình tài chính, thói quan chi tiêu của 2 nhóm khách hàng. Từ đó xây dựng định hướng phát triển sản phẩm phù hợp đối với nhu cầu từng nhóm khách hàng

<details>
<summary><b>Xem toàn bộ code </b></summary>
    
```
# Tạo các nhóm DTI từ credit_card_users
high_dti_credit = credit_card_users[credit_card_users['dti_group'] == 'Cao (>0.4)']
very_low_dti_credit = credit_card_users[credit_card_users['dti_group'] == 'Rất thấp (0-0.2)']

print(f"Số lượng khách hàng DTI cao (Credit): {len(high_dti_credit):,}")
print(f"Số lượng khách hàng DTI rất thấp (Credit): {len(very_low_dti_credit):,}")

# Tạo heatmap phân bố độ tuổi và thu nhập
plt.figure(figsize=(18, 12))

# Heatmap 1: Nhóm DTI cao - Credit Card Users
plt.subplot(2, 3, 1)
age_income_high_dti_credit = pd.crosstab(high_dti_credit['age_group'], high_dti_credit['group_income'])
sns.heatmap(age_income_high_dti_credit, annot=True, fmt='d', cmap='Reds', cbar_kws={'label': 'Số lượng KH'})
plt.title('PHÂN BỐ ĐỘ TUỔI - THU NHẬP\nNHÓM DTI CAO (>0.4) - CREDIT CARD', fontsize=12, fontweight='bold')
plt.xlabel('Nhóm thu nhập')
plt.ylabel('Độ tuổi')

# Heatmap 2: Nhóm DTI rất thấp - Credit Card Users
plt.subplot(2, 3, 2)
age_income_very_low_dti_credit = pd.crosstab(very_low_dti_credit['age_group'], very_low_dti_credit['group_income'])
sns.heatmap(age_income_very_low_dti_credit, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Số lượng KH'})
plt.title('PHÂN BỐ ĐỘ TUỔI - THU NHẬP\nNHÓM DTI RẤT THẤP (0-0.2) - CREDIT CARD', fontsize=12, fontweight='bold')
plt.xlabel('Nhóm thu nhập')
plt.ylabel('Độ tuổi')

```
</details>

![i8](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/age_income_dti_ct.png)

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
# Tìm top 3 category lớn nhất trong credit card users
top3_categories = credit_card_users['top_category'].value_counts().head(3).index.tolist()

# Lọc chỉ top 3 category cho cả hai nhóm
top3_data_high_dti = high_dti_credit[high_dti_credit['top_category'].isin(top3_categories)]
top3_data_low_dti = very_low_dti_credit[very_low_dti_credit['top_category'].isin(top3_categories)]

# Tính chi tiêu trung bình theo độ tuổi và category cho cả hai nhóm
age_category_spent_high_dti = top3_data_high_dti.groupby(['age_group', 'top_category'])['total_spent'].sum().unstack()
age_category_spent_low_dti = top3_data_low_dti.groupby(['age_group', 'top_category'])['total_spent'].sum().unstack()

# Vẽ biểu đồ so sánh
plt.figure(figsize=(15, 12))

# Màu sắc cố định cho 3 category
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# Nhóm DTI cao
plt.subplot(1, 2, 1)
x = np.arange(len(age_category_spent_high_dti.index))
width = 0.25

for i, category in enumerate(top3_categories):
    plt.bar(x + i*width - width, age_category_spent_high_dti[category], width, 
            label=category, color=colors[i], alpha=0.8)

plt.xlabel('Nhóm tuổi', fontsize=12)
plt.ylabel('Tổng chi tiêu trung bình (Triệu $)', fontsize=12)
plt.title('Tổng chi tiêu top 3 ngành hàng theo độ tuổi \n NHÓM DTI CAO (>0.4)', 
          fontsize=12, fontweight='bold')
plt.xticks(x, age_category_spent_high_dti.index)
plt.grid(axis='y', alpha=0.3)

# Nhóm DTI rất thấp
plt.subplot(1, 2, 2)
for i, category in enumerate(top3_categories):
    plt.bar(x + i*width - width, age_category_spent_low_dti[category], width, 
            label=category, color=colors[i], alpha=0.8)

plt.xlabel('Nhóm tuổi', fontsize=12)
plt.ylabel('Tổng chi tiêu trung bình (Triệu $)', fontsize=12)
plt.title('Tổng chi tiêu top 3 ngành hàng theo độ tuổi \n NHÓM DTI RẤT THẤP (0-0.2)', 
          fontsize=12, fontweight='bold')
plt.xticks(x, age_category_spent_low_dti.index)
plt.grid(axis='y', alpha=0.3)

# Tạo chú thích chung cho cả 2 biểu đồ
plt.figlegend(handles=[plt.Rectangle((0,0),1,1, fc=colors[i]) for i in range(len(top3_categories))],
            labels=top3_categories,
            title='Danh mục chi tiêu',
            loc='upper center',
            bbox_to_anchor=(0.5, 0.05),
            ncol=3,
            fontsize=11,
            frameon=True,
            fancybox=True,
            shadow=True)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
```
</details>

![i9](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/categories.png)
![i10](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/top3_category_dti.png)

*Nhận xét:*
- Có thể thấy nhóm khách hàng có tỷ lệ DTI cao chủ yếu có mức thu nhập dưới 50k$ và một số ít tập trung ở 30k$-75k$. Xét ở khía cạnh tổng nợ, nhóm này khách hàng ở độ tuổi 35-54 đang có tổng nợ ở mức trung bình - cao, còn lại nhóm tuổi 25-34, 55+ có nợ ở mức trung bình - thấp. Đối với nhóm DTI rất thấp, chủ yếu những khách hàng nằm ở nhóm này là người cao tuổi có thu nhập thấp, nợ của họ cũng rất ít, hầu hết chỉ dưới 5k$.
- Tỷ lệ dịch vụ được chi tiêu của nhóm sử dụng thẻ Credit khá tương đồng với tổng thể, không có nhiều khác biệt trong hành vi chi tiêu của nhóm này. Có thể thấy trong nhóm khách hàng có chỉ số DTI cao, chi tiêu cao nhất được thể hiện ở nhóm 35-54 tuổi và thấp nhất ở nhóm 25-34 và 65+. Trong khi đó, nhóm DTI rất thấp với phần lớn khách hàng là người cao tuổi thu nhập thấp, nợ ít chủ yếu tiêu dùng cho Ăn uống.

*Tôi sẽ chia khách hàng nhóm DTI cao và thấp theo tổng chi tiêu và phân bố tổng nợ trung bình thành 3 nhóm như sau, với các hướng phát triển khác nhau:*

* Nhóm chi tiêu cao, nợ cao (35-54): Đây là nhóm dù có nợ cao, nhưng sức mua tốt, cần thúc đấy tiêu dùng những sản phẩm tín dụng an toàn: chính sách hoàn tiền khi chi tiêu các mặt hàng phổ biến (Ăn uống, Xăng dầu & Phương tiện), các gói dịch vụ Du lịch trả góp 0%,...
* Nhóm chi tiêu thấp, nợ cao (25-34 và 55-64): Đây là nhóm khách hàng đang ngại chi tiêu, sợ nợ, cần thúc đẩy những dịch vụ tái cấu trúc nợ, các khoản vay ưu đãi kéo dài kỳ hạn, lãi suất thấp, các khoản vay tái xây dựng tín dụng (credit builder loans)
* Nhóm chi tiêu thấp, nợ thấp (65+): Đây là nhóm đối tượng khách hàng có tiểm năng tăng trưởng tín dụng cao và an toàn nhất, có thể đề xuất nhóm sản phẩm ưu đãi vay tiêu dùng đặc biệt là cho vay tiêu dùng đối với các dịch vụ thuộc nhóm Du lịch & Giải trí (vay du lịch, ưu đãi bảo hiểm du lịch, thẻ đồng thương hiệu khách sạn/hàng không,...)


# Phân tích tỷ lệ sử dụng hạn mức và điểm tín dụng của các nhóm khách hàng để đánh giá rủi ro: 

- Ở phần này, tôi phân tích biến điểm tín dụng và tỷ lệ sử dụng hạn mức tín dụng để xem xét rủi ro, xây dựng điều kiện sử dụng thêm cho các hướng sản phẩm đối với từng nhóm khách hàng

<details>
<summary><b>Xem toàn bộ code</b></summary>
  
```
plt.figure(figsize=(16, 8))  

# Phân bố điểm tín dụng khách hàng dùng thẻ credit có DTI cao
plt.subplot(1, 2, 1)
high_dti_crosstab = pd.crosstab(high_dti_credit['age_group'], high_dti_credit['credit_score_group'])
high_dti_crosstab = high_dti_crosstab.reindex(columns=credit_labels)

bar1 = high_dti_crosstab.plot(kind='bar', color=credit_colors, ax=plt.gca(), legend=False)
plt.title('PHÂN BỐ ĐIỂM TÍN DỤNG THEO ĐỘ TUỔI\nKHÁCH HÀNG THẺ TÍN DỤNG - DTI CAO (>0.4)', fontsize=12, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Số lượng khách hàng')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Phân bố điểm tín dụng khách hàng dùng thẻ credit có DTI rất thấp
plt.subplot(1, 2, 2)
very_low_dti_crosstab = pd.crosstab(very_low_dti_credit['age_group'], very_low_dti_credit['credit_score_group'])
very_low_dti_crosstab = very_low_dti_crosstab.reindex(columns=credit_labels)

bar2 = very_low_dti_crosstab.plot(kind='bar', color=credit_colors, ax=plt.gca(), legend=False)
plt.title('PHÂN BỐ ĐIỂM TÍN DỤNG THEO ĐỘ TUỔI\nKHÁCH HÀNG THẺ TÍN DỤNG - DTI RẤT THẤP (0-0.2)', fontsize=12, fontweight='bold')
plt.xlabel('Độ tuổi')
plt.ylabel('Số lượng khách hàng')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)


plt.figlegend(handles=[plt.Rectangle((0,0),1,1, fc=color) for color in credit_colors],
            labels=credit_labels,
            title='Nhóm điểm tín dụng',
            loc='upper center', 
            bbox_to_anchor=(0.5, 0.98),  
            fontsize=11,
            frameon=True,
            fancybox=True)

plt.tight_layout(rect=[0, 0, 1, 0.88])  
plt.show()
```
</details>

![i11](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/credit_score_dti_ct.png)
![i11](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/stat_credit_score.png)

<details>
<summary><b>Xem toàn bộ code</b></summary>
    
```
plt.figure(figsize=(18, 10))

# Loại bỏ khách hàng không có credit limit
high_dti_credit = high_dti_credit[high_dti_credit['credit_limit'] > 0].copy()
very_low_dti_credit = very_low_dti_credit[very_low_dti_credit['credit_limit'] > 0].copy()
credit_card_users = credit_card_users[credit_card_users['credit_limit'] > 0].copy()

utilization_data = [
    high_dti_credit['credit_utilization_ratio'].values,  
    very_low_dti_credit['credit_utilization_ratio'].values  
]
labels = ['DTI Cao (>0.36)', 'DTI Rất thấp (0-0.2)']

# Tính toán tỷ lệ sử dụng hạn mức
high_dti_credit['credit_utilization_ratio'] = high_dti_credit['total_spent'] / high_dti_credit['credit_limit']
very_low_dti_credit['credit_utilization_ratio'] = very_low_dti_credit['total_spent'] / very_low_dti_credit['credit_limit']
credit_card_users['credit_utilization_ratio'] = credit_card_users['total_spent'] / credit_card_users['credit_limit']

# Phân nhóm sử dụng hạn mức
def categorize_utilization(ratio):
    if ratio <= 0.3:
        return 'Thấp (0-30%)'
    elif ratio <= 0.7:
        return 'Trung bình (30-70%)'
    elif ratio <= 1:
        return 'Cao (70-100%)'
    else:
        return 'Vượt hạn mức (>100%)'

credit_card_users['utilization_group'] = credit_card_users['credit_utilization_ratio'].apply(categorize_utilization)
high_dti_credit['utilization_group'] = high_dti_credit['credit_utilization_ratio'].apply(categorize_utilization)
very_low_dti_credit['utilization_group'] = very_low_dti_credit['credit_utilization_ratio'].apply(categorize_utilization)

age_util_low_dti = pd.crosstab(
    very_low_dti_credit['age_group'],
    very_low_dti_credit['utilization_group'],
    normalize='index'
) * 100

age_util_high_dti = pd.crosstab(
    high_dti_credit['age_group'],
    high_dti_credit['utilization_group'],
    normalize='index'
) * 100

# Vẽ boxplot 
plt.subplot(1, 3, 1)
box_plot = plt.boxplot(
    utilization_data,
    labels=labels,
    showfliers=False,     
    patch_artist=True,
    whis=[5, 96]      
)

colors = ['#FF6B6B', '#4ECDC4']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title("Phân bố tỷ lệ sử dụng hạn mức\n(Khách hàng dùng thẻ Credit)", 
          fontsize=13, fontweight='bold')
plt.ylabel("Tỷ lệ sử dụng hạn mức")
plt.grid(axis='y', alpha=0.3)

colors = ['#2E8B57', '#FFD700', '#FF6347', '#8B0000']

# Phân bố hạn mức nhóm DTI rất thấp
plt.subplot(1, 3, 2)
age_util_low_dti.plot(kind='bar', stacked=True, color=colors, ax=plt.gca())
plt.title("Phân bố mức sử dụng hạn mức theo nhóm tuổi\nNhóm DTI rất thấp (0–0.2)", fontsize=13, fontweight='bold')
plt.xlabel("Nhóm tuổi")
plt.ylabel("Tỷ lệ (%)")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.legend(title="Mức sử dụng hạn mức", bbox_to_anchor=(1.05, 1), loc='upper left')

# Phân bố hạn mức nhóm DTI cao
plt.subplot(1, 3, 3)
age_util_high_dti.plot(kind='bar', stacked=True, color=colors, ax=plt.gca())
plt.title("Phân bố mức sử dụng hạn mức theo nhóm tuổi\nNhóm DTI cao (>0.36)", fontsize=13, fontweight='bold')
plt.xlabel("Nhóm tuổi")
plt.ylabel("Tỷ lệ (%)")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.legend(title="Mức sử dụng hạn mức", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Tính tỷ lệ khách hàng vượt hạn mức
high_dti_over_limit = (high_dti_credit['credit_utilization_ratio'] > 1).sum()
very_low_dti_over_limit = (very_low_dti_credit['credit_utilization_ratio'] > 1).sum()

print(f"\n=== TỶ LỆ VƯỢT HẠN MỨC (>100%) ===")
print(f"DTI Cao: {high_dti_over_limit}/{len(high_dti_credit)} ({high_dti_over_limit/len(high_dti_credit)*100:.1f}%)")
print(f"DTI Rất thấp: {very_low_dti_over_limit}/{len(very_low_dti_credit)} ({very_low_dti_over_limit/len(very_low_dti_credit)*100:.1f}%)")

```
</details>

![i13](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/credit_limit_dti_ct.png)
![i14](https://github.com/kiettran13/Customer_analysis/blob/main/Chart/stat_credit_limit.png)

*Nhận xét:*
- Nhóm khách hàng có DTI cao đang chịu mức áp lực điểm tín dụng lớn và có tỷ lệ vượt hạn mức cao và vượt giới hạn cao, đặc biệt ở nhóm tuổi 35-54, vì vậy để phòng tránh rủi ro vỡ nợ tín dụng của khách hàng, các sản phẩm cho nhóm khách hàng này nên có điều kiện sử dụng (ví dụ chỉ áp dụng cho khách hàng có điểm tín dụng từ trung bình đến cao)
- Nhóm khách hàng có DTI thấp trung bình có tỷ lệ dùng hạn mức cao hơn nhóm DTI thấp và có điểm tín dụng ở mức trung bình - cao chiếm tới gần 92%, có thể xem xét mở rộng hạn mức cho nhóm khách hàng này.


# Kết luận dự án
*Dựa vào những thông tin ở trên, dưới đây là kết luận cho 3 hướng phát triển sản phẩm phù hợp cho 3 nhóm khách hàng khác nhau*
## Nhóm khách hàng thu nhập thấp, nợ nhiều – chi tiêu mạnh (chủ yếu 35–54 tuổi):
* Đây là phân khúc có sức mua cao và đang đóng góp phần lớn doanh thu phí hiện tại. Dù gánh nợ cao, họ vẫn duy trì chi tiêu ổn định ở các ngành thiết yếu và đặc biệt là Travel (đỉnh cao ở nhóm 45–54), nhóm khách hàng này cần được kích cầu thay vì kìm hãm. Hướng phát triển trọng tâm là các sản phẩm kích thích chi tiêu an toàn: cashback cao 7–10% cho Ăn uống, Xăng dầu & Phương tiện, chương trình trả góp 0% lãi 12–24 tháng cho Du lịch cùng chính sách tăng hạn mức nhẹ. Để kiểm soát rủi ro, các ưu đãi chỉ áp dụng cho khách có điểm tín dụng từ trung bình đến cao (> 680) và không có lịch sử trả trễ trong 12 tháng gần nhất. Với nhóm này, mục tiêu không phải giảm nợ mà là giúp họ tiêu nhiều hơn, từ đó gia tăng doanh thu phí và lãi quay vòng một cách bền vững.
## Nhóm khách hàng thu nhập thấp, nợ nhiều – chi tiêu yếu (25–34 tuổi và 55–64 tuổi):
* Đây là nhóm đang thực sự khó khăn, điểm tín dụng thấp, chi tiêu bị cắt giảm mạnh, chỉ còn các khoản thiết yếu, và có nguy cơ cao rơi vào nợ xấu hoặc rời ngân hàng. Nếu không can thiệp kịp thời, tổ chức sẽ mất trắng một lượng lớn khách hàng trung thành. Hướng phát triển phù hợp và cấp bách nhất là cứu nợ và xây dựng lại niềm tin: Triển khai các gói dịch vụ tái cấu trúc nợ và cấp thẻ tín dụng chuyển đổi số dư 0% lãi dài hạn, cung cấp các khoản nợ mới kéo dài kỳ hạn trả nợ tối đa 36–48 tháng với lãi suất ưu đãi hoặc các khoản vay tái xây dựng tín dụng (Credit Builder Loan). Mục tiêu là đưa họ từ trạng thái “không dám tiêu” trở lại thành khách hàng chi tiêu bình thường trong thời gian tới.
## Nhóm khách hàng thu nhập thấp, nợ ít - chi tiêu yếu (chủ yếu 65+):
* Nhóm này, sở hữu điểm tín dụng từ trung bình đến cao (chỉ khoảng 10% có điểm tín dụng thấp). Dù tình trạng tài chính rất khỏe mạnh, họ vẫn duy trì tỷ lệ sử dụng hạn mức ở mức trung bình và cao (trung bình ~40%), cho thấy vẫn có nhu cầu tín dụng thường xuyên. Tuy nhiên, cơ cấu chi tiêu của nhóm này cũng thiên về các khoản thiết yếu như Ăn uống. Đây chính là nhóm khách hàng có dư địa tăng trưởng tín dụng lớn nhất, hoàn toàn đủ khả năng hấp thụ thêm hạn mức và các khoản vay mới. Đây là nhóm đối tượng khách hàng có tiểm năng tăng trưởng tín dụng, có thể đề xuất nhóm sản phẩm ưu đãi vay tiêu dùng đặc biệt là cho vay tiêu dùng đối với các dịch vụ thuộc nhóm Du lịch & Giải trí (vay du lịch, ưu đãi bảo hiểm du lịch, thẻ đồng thương hiệu khách sạn/hàng không,...) nhằm kích hoạt nhu cầu chi tiêu cao cấp và gia tăng mạnh danh thu từ nhóm này


















