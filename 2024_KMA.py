#!/usr/bin/env python
# coding: utf-8

# # 데이터 준비
# - 제공받은 날씨 데이터와 화재 데이터를 일자와 지역 구분1, 2를 기준으로 left join한 데이터프레임 활용

# In[ ]:


# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import missingno as msno


# In[ ]:


# 결측치 처리 하기 이전 데이터 가져오기
df_b1 = pd.read_csv("data1.csv")
df_b2 = pd.read_csv("data2.csv")
df_b3 = pd.read_csv("data3.csv")
df_b4 = pd.read_csv("data4.csv")

# 데이터 다시 합치기
df_before = pd.concat([df_b1, df_b2, df_b3, df_b4])
df_before


# # EDA

# In[ ]:


# 데이터 정보 파악
df_before.info()


# In[ ]:


# 데이터 기초통계량 확인_기상정보 위주
weather_feature = df_before.describe().iloc[:,2:13]
weather_feature


# In[ ]:


# 기상 정보의 분포 확인
df_before[weather_feature.columns.tolist()].hist()


# - rn_day와 ws_min의 경우 분포가 과하게 왼쪽에 치우친 모양 관찰
# -> 피처의 유의미한 차이 관찰하기 어려움

# ## 결측치 처리
# - 공공데이터를 통해 처리할 수 있는 공간은 처리하고 그 외에는 인접한 지역의 데이터로 처리
# - 이 후 발생한 약 2만개의 결측치를 포함한 행은 dropna()

# In[ ]:


# 결측치 처리 이전 데이터 결측치 분포 확인
msno.matrix(df_before)


# In[ ]:


# 공공 데이터와 공간 보간을 활용해 결측치 처리
## 예시 : 전라북도 해안 일대를 인접한 지역인 전라북도의 부안 공공데이터로 결측치 처리

# 부안 풍속 데이터 가져오기
buan = pd.read_csv(("buan_wind.csv"), encoding='cp949', skiprows=13)
buan = buan.rename(columns={'일시': 'date', '평균풍속(m/s)' : 'ws_mean', '최대풍속(m/s)' : 'ws_max', '최대순간풍속(m/s)' : 'ws_ins_max'})
buan_ws = buan[['date', 'ws_mean', 'ws_max', 'ws_ins_max']]
buan_ws


# In[ ]:


# 전북특별자치도와 특정 지역에 해당하는 조건 설정
condition = (df_before['district_1'] == '전북특별자치도') & (df_before['district_2'].isin(['군산시', '김제시', '부안군', '고창군', '정읍시']))

# 처리 이전 결측치 개수 출력
print('처리 이전 결측치 개수: ', df_before[condition]['ws_mean'].isna().sum())

if condition.any():
    # 조건에 해당하는 행을 복사하여 새로운 데이터프레임 생성
    df_filtered = df_before[condition].copy()

    # 조건에 해당하는 행과 buan_ws 데이터프레임을 날짜(date) 기준으로 병합
    df_filtered = df_filtered.merge(buan_ws, on='date', how='left', suffixes=('', '_ref'))

    # 결측치 대체
    for i in ['ws_mean', 'ws_max', 'ws_ins_max']:
        df_filtered[i] = df_filtered[i].fillna(df_filtered[f'{i}_ref'])

    # 병합 결과에서 사용된 보조 열 제거
    df_filtered = df_filtered.drop(['ws_mean_ref', 'ws_max_ref', 'ws_ins_max_ref'], axis=1)

    # 원본 데이터프레임에서 조건에 해당하는 행 제거 후 결측치 대체된 행 추가
    df_before = pd.concat([df_before[~condition], df_filtered])

    # 병합 후 원본 데이터프레임과 일치하도록 condition을 다시 정의
    condition = (df_before['district_1'] == '전북특별자치도') & (df_before['district_2'].isin(['군산시', '김제시', '부안군', '고창군', '정읍시']))

# 처리한 이후의 결측치 개수 출력
print('처리 이후 결측치 개수 :', df_before[condition]['ws_mean'].isna().sum())


# In[ ]:


# 결과 데이터 가져오기
df_1 = pd.read_csv("df_all_1.csv")
df_2 = pd.read_csv("df_all_2.csv")
df_3 = pd.read_csv("df_all_3.csv")
df_4 = pd.read_csv("df_all_4.csv")


# In[ ]:


# 데이터 다시 합치기
df = pd.concat([df_1, df_2, df_3, df_4])
# 처리한 결측치 확인
msno.matrix(df)


# In[ ]:


# 지역별로 구분한 데이터 가져오기
import pandas as pd
inland1 = pd.read_csv("inland_some2.csv")
inland2 = pd.read_csv("inland_some.csv")
seohae = pd.read_csv("seohae.csv")
sea_mt = pd.read_csv("taebaek_mountain.csv")
in_mt = pd.read_csv("inland_mountain.csv")
namhae = pd.read_csv("namhaean.csv")


# ## 지역별 발생 요인의 차이

# In[ ]:


import pandas as pd

# 데이터프레임 리스트 및 해당 이름 리스트 생성
dataframes = [inland1, inland2, sea_mt, in_mt, seohae, namhae]
names = ['Inland 1', 'Inland 2', 'Sea & Mountain', 'Inland & Mountain', 'Seohae', 'Namhae']

# 결과를 저장할 리스트
factor_list = []

# ignition_factor_category_2의 비율을 구하는 함수
def factor(df, name):
    df_t = df[['district_1', 'district_2', 'ta_max', 'ta_min', 'ta_max_min', 'rn_day', 'ws_max',
               'ws_ins_max', 'ws_mean', 'ws_min', 'hm_max', 'hm_mean', 'hm_min',
               'date', 'fire_type_1', 'fire_type_2', 'ignition_factor_category_1',
               'ignition_factor_category_2', 'casualties', 'dead', 'injury',
               'property_damage', 'location_category_1', 'location_category_2', 'location_category_3']]

    df_t = df_t.dropna(subset=['dead'])
    df_factor = df_t.value_counts('ignition_factor_category_2', normalize=True).reset_index(name='prop(%)').nlargest(10, 'prop(%)')
    df_factor['prop(%)'] = df_factor['prop(%)'].round(4) * 100
    df_factor = df_factor.rename(columns={'ignition_factor_category_2': f'{name} 화재 요인'})
    factor_list.append(df_factor)

# 지역별 함수 적용
for df, name in zip(dataframes, names):
    factor(df, name)

# 동시에 비교
df_cat = pd.concat(factor_list, axis=1)
df_cat


# # 데이터 분석

# ## 지역별 카이제곱을 통한 피처의 영향도 관찰

# In[ ]:


# 데이터 범주화
def chi(df) :
    weather_features = ['ta_max','ta_min', 'ta_max_min', 'ws_max', 'ws_ins_max', 'ws_mean', 'hm_max', 'hm_mean', 'hm_min']
    for i in weather_features :
        df[f'{i}_cut'] = pd.qcut(df[i], 5, labels = [1,2,3,4,5])

    # 화재 발생여부에 따라 data 분할
    set1 = df.query('not ym.isna()')
    set2 = df.query('ym.isna()')

    # 범주화한 피처 기준으로 그룹화
    cut_features = ['ta_max_cut', 'ta_min_cut','ta_max_min_cut', 'ws_max_cut', 'ws_ins_max_cut', 'ws_mean_cut', 'hm_max_cut', 'hm_mean_cut', 'hm_min_cut']

    cut_df = pd.DataFrame({'grade' :['매우낮음', '낮음', '보통', '높음', '매우 높음']})

    for i in cut_features :
        for_set_1 = set1.groupby(i).size().reset_index(name='size')
        for_set_2 = set2.groupby(i).size().reset_index(name='size')
        cut_df[f'{i}_fire O'] = for_set_1['size']
        cut_df[f'{i}_fire X'] = for_set_2['size']

    # 카이제곱 결과 가져오기
    from scipy.stats import chi2_contingency

    p_value_list = []

    for i in range(1,19,2):
        for_chi = cut_df.iloc[:, i : i+2]
        chi2_stat, p_val, dof, expected = chi2_contingency(for_chi)
        p_value_list.append(p_val)
    return p_value_list


# In[ ]:


# 지역별 카이제곱 결과 불러오기
chi_result = pd.DataFrame({'피처' : ['최대기온', '최저기온', '일교차', '최대풍속', '최대순간풍속', '평균풍속', '최대습도', '평균습도', '최저습도'],
                           '서울/광주/대구' : chi(inland1),
                           '대전/세종/충남' : chi(inland2),
                           '태백산맥~동해안' : chi(sea_mt),
                           '내륙산간' : chi(in_mt),
                           '서해안' : chi(seohae),
                           '남해안' : chi(namhae)})
chi_result.set_index('피처', inplace = True)
chi_result


# ## 지역별 유의미한 피처 시각화

# In[ ]:


fig, ax = plt.subplots(2,3, figsize = (15,10))

df_list = [inland1, inland2, sea_mt, in_mt, seohae, namhae]
titles = ['Inland 1', 'Inland 2', 'Sea & Mountain', 'Inland & Mountain', 'Seohae', 'Namhae']
best_feature = 'hm_mean_cut'

for i, (df, title) in enumerate(zip(df_list, titles)) :
    df = df.copy()
    df['fire'] = np.where(df['ym'].isna(), 'fire x', 'fire o')
    df_group1 = df.query('fire=="fire o"').groupby(best_feature).size().reset_index(name='O')
    df_group2 = df.query('fire=="fire x"').groupby(best_feature).size().reset_index(name='x')
    merged_df = df_group1.merge(df_group2)
    merged_df['ratio'] = round(merged_df['O'] / merged_df.iloc[:, 1:3].sum(axis=1),4)
    ax = ax.flatten()
    sns.barplot(x = best_feature, y='ratio', data = merged_df, ax = ax[i])
    ax[i].set_title(title)
    ax[i].grid(True)


# ## xgboost를 활용한 피처 관찰

# In[ ]:


# 라이브러리 불러오기
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def xgboost(df) :
  # 화재 컬럼 생성
    df_t = df.copy()
    df_t['fire'] = np.where(df['ym'].isna(), 0, 1)

  # 피처 정리
    X = df_t[['ta_max', 'ta_min', 'ta_max_min', 'ws_max',
              'ws_ins_max', 'ws_mean','hm_max', 'hm_mean', 'hm_min']]
    y = df_t['fire']

  # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

  # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2, random_state=111)

  # 하이퍼 파리미터 지정
    params = {'n_estimators': [100, 200, 300, 400, 500],
              'max_depth': [7, 8, 9, 10],
              'learning_rate': [0.01, 0.1],
              'lambda': [0.1, 0.5, 1, 2, 3]}

  # 모델 생성
    model = xgb.XGBClassifier(random_state=42)

  # 하이퍼 파라미터 튜닝
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model, params, n_jobs=-1, cv = 3, scoring='accuracy')
    gs.fit(X_train, y_train)

  # 베스트 모델 선정
    best_model = xgb.XGBClassifier(**gs.best_params_)
    best_model.fit(X_train, y_train)
    return best_model


# In[ ]:


# 내륙 지역(서울, 대구, 광주, 대전, 층남, 세종) 통합 피처 중요도 관찰
inland = pd.concat([inland1, inland2])
model = xgboost(inland)
Inland_ax = xgb.plot_importance(model)
Inland_ax


# In[ ]:


# 동해안 ~ 태백산맥 지역 피처 중요도 관찰
model = xgboost(sea_mt)
Inland_ax = xgb.plot_importance(model)
Inland_ax


# In[ ]:


# 내륙산간 지역 피처 중요도 관찰
model = xgboost(in_mt)
Inland_ax = xgb.plot_importance(model)
Inland_ax


# In[ ]:


# 서해안 지역 피처 중요도 관찰
model = xgboost(seohae)
Inland_ax = xgb.plot_importance(model)
Inland_ax


# In[ ]:


# 남해안 지역 피처 중요도 관찰
model = xgboost(namhae)
Inland_ax = xgb.plot_importance(model)
Inland_ax


# In[ ]:




