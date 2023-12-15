import os
import pandas as pd
path = "data/magnatagatune/annotations_final.csv"


def df_clear(df):
    header = df.columns[0].split("\t")
    header_list = [item.replace('"', '') for item in header]

    columns = df.columns
    data_list = []
    for idx in range(len(df)):
        data = df[columns[0]].iloc[idx].split("\t")
        data_tmp = [item for item in data]
        data_list.append(data_tmp)

    df = pd.DataFrame(data_list, columns=header_list)

    for col in df.columns[1:-1]:
        df[col] = df[col].str.replace('"', '').astype(int)
    return df

def df_tag(df, dic):
    cols = dic.keys()
    for col in cols:
        origin = dic[col]
        df[col] = df[origin].max(axis=1)
    return df

def tag_cnt(df):
    # 각 열에서 1의 개수를 계산
    count_ones = (df == 1).sum(axis=0)

    print(count_ones)
    # 새로운 행으로 df에 추가
    d1f = count_ones.to_frame()
    d1f.to_csv("tag_count.csv")

# def make_origin(df):
#     columns = df.columns
#     line_list = []
#     line = ''
#     for col in columns[:-1]:
#         line += f'\"{col}\"\t'
#     line += columns[-1]
#     line_list.append(line)

#     for row in df:

df = pd.read_csv(path)
df_new = df_clear(df)
df_new.to_csv('df_new.csv', index=False)


tag = ['strings', 'string', 'guitar', 'plucking', 'sitar','violins', 'bass', 'violin', 'harp', 'fiddle', 'lute', 'viola', 'cello']
# real_tag = {strings  string  guitar  plucking  sitar  violins  bass  violin  harp  fiddle  lute  viola  cello}
df_do = df_new[tag]
df_strings = pd.DataFrame()
df_strings['string'] = df_do[['strings', 'string']].max(axis=1)
df_strings['instrument'] = df_do[['guitar', 'plucking', 'sitar','violins', 'bass', 'violin', 'harp', 'fiddle', 'lute', 'viola', 'cello']].max(axis=1)

df_strings['same'] = df_strings['instrument']
count_b = len(df_strings[(df_strings['string'] >= 1) & (df_strings['instrument'] >= 1)])
print(count_b)

"""
count_i = 0
if df_strings['string']<1 and df_strings['instrument'] >=1:
    count_i+=1

count_s = 0
if df_strings['string']>=1 and df_strings['instrument'] <1:
    count_s+=1

print(f"both: {count_b}, only ins: {count_i}, only_s:{count_s}")



cols = [f'"{x}"' for x in cols]
header = '\t'.join(cols)

data = []
for i in range(len(df_tmp)):
    row = list(df_tmp.iloc[i])
    row_tmp = [f'"{x}"' for x in row[:-1]]
    row_tmp.append(row[-1])
    string = '\t'.join(row_tmp)
    data.append(string)

result = pd.DataFrame(data=data, columns = [header])


result.to_csv("tag123.csv", index=False)

"""