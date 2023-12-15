import os
import pandas as pd
# path = "data/magnatagatune/annotations_final.csv"
path = "df_new_merged.csv"


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

df_do = pd.read_csv(path)
# df_new = df_clear(df)
# df_new.to_csv('df_new.csv', index=False)


#tag = ['clip_id', 'strings', 'string','guitar', 'plucking', 'sitar','violins', 'bass', 'violin', 'harp', 'fiddle', 'lute', 'viola', 'cello', "guitars", "acoustic guitar", "electric guitar", "classical guitar"]
#tag = ['clip_id','keyboard', 'organ', 'synthesizer', 'piano', 'piano solo']
# tag = ['clip_id', "clarinet","woodwind","horns","wind","trumpet","horn","oboe","flutes","sax"]
#tag = ['clip_id', "banjo", "percussion", "drum", "drums", "bongos"]
# real_tag = {strings  string  guitar  plucking  sitar  violins  bass  violin  harp  fiddle  lute  viola  cello}
# df_do = df_new[tag]

df_strings = pd.DataFrame()
df_strings['clip_id'] = df_do[['clip_id']]
#df_strings['string'] = df_do[['strings', 'string']].sum(axis=1)
#df_strings['keyboard'] = df_do[['keyboard']].sum(axis=1)
#df_strings['percussion'] = df_do[['percussion']].sum(axis=1)
df_strings['voice'] = df_do[['no vocal', 'vocal']].sum(axis=1)
df_strings['sex'] = df_do[['male', 'female', 'duet']].sum(axis=1)
# df_strings['instrument'] = df_do[["clarinet","horns","trumpet","horn","oboe","flutes","sax"]].sum(axis=1)
#df_strings['instrument'] = df_do[['clip_id', 'strings', 'string','guitar', 'plucking', 'sitar','violins', 'bass', 'violin', 'harp', 'fiddle', 'lute', 'viola', 'cello', "guitars", "acoustic guitar", "electric guitar", "classical guitar"]].sum(axis=1)
#df_strings['instrument'] = df_do[['organ', 'synthesizer', 'piano', 'piano solo']].sum(axis=1)
#df_strings['instrument'] = df_do[["banjo", "drum", "drums", "bongos"]].sum(axis=1)

#count_b = df_strings(df_strings['string']>=1 and df_strings['instrument'] >=1)

# count_b = len(df_strings[(df_strings['wind'] >= 1) & (df_strings['instrument'] >= 1)])


# count_i = len(df_strings[(df_strings['wind'] < 1) & (df_strings['instrument'] >= 1)])

# # strings 또는 string만 있는 애들
# count_s = len(df_strings[(df_strings['wind'] >= 1) & (df_strings['instrument'] < 1)])

condition = (df_strings['voice'] >= 2)
cond = (df_strings['sex'] == 0)
print(condition)
print(cond)
s_idx = df_do.loc[condition, 'clip_id'] # only string
# s_idx = df_do.loc[condition].index # only string

#condition = (df_do['organ']>=1)
#s_idx = df_do.loc[condition, 'clip_id']

# print(f"both: {count_b}, only ins: {count_i}, only_s:{count_s}")
print(s_idx)

text = '\n'.join(map(str, s_idx))

# 파일에 쓰기
with open('new_vocal_sex_result2.txt', 'w') as file:
    file.write(text)

"""


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