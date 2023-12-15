import pandas as pd

path = "/home/seunghoi/clmr/20231124_after_tag141_v2.csv"
merged_path = "/home/seunghoi/clmr/20231124_after_tag141_v3.csv"
# changes_path = "/home/seunghoi/clmr/changes.txt"

def update_parent_remove_children(df, parent_tag, child_tags):
    # 부모 태그와 자식 태그 확인
    if parent_tag not in df.columns or not all(tag in df.columns for tag in child_tags):
        raise ValueError(f"Parent or child tag(s) not found in the dataframe for '{parent_tag}'")
    
    # 변경 사항 추적을 위한 집합
    changed_indices = set()
    
    # 자식 태그가 1이고 부모 태그가 0인 행 찾기
    for child_tag in child_tags:
        mask = (df[child_tag] == 1) & (df[parent_tag] == 0)
        changed_indices.update(df[mask].index)
        df.loc[mask, parent_tag] = 1
    
    # 자식 태그 열 삭제
    df.drop(columns=child_tags, inplace=True)
    
    # 업데이트된 데이터프레임과 변경된 인덱스 반환
    return df, changed_indices

df_new = pd.read_csv(path)

# tag_pairs = [ # 첫번째에 통합할 부모태그이름 적고 그 뒤엔 없앨 애들 태그 이름 적기
#     ('piano', ['piano solo']),
#     ('guitar', ['guitars']),
#     ('drum', ['drums']),
#     ('string', ['strings']),
#     ('violin', ['violins']),
#     ('wind', ['woodwind']),
#     ('no vocal', ['no voice', 'no voices', 'no vocals', 'no singer', 'no singing']),
#     ('vocal', ['singer', 'singing', 'voices', 'solo', 'vocals','voice']),
#     ('female', ['female singing', 'female voice', 'female vocal', 'woman singing', 'woman', 'female singer', 'women', 'girl', 'female opera', 'female vocals']),
#     ('male', ['male vocal', 'men', 'male voice', 'male singer', 'man singing', 'male vocals', 'man']),
#     ('classic',['clasical', 'classical'])
# ]
tag_pairs = [
    ('fast', ['fast beat']),    
]

# remove_tag = ['beat','no beat', 'not rock', 'not opera', 'not english', 'plucking', 'guitar', 'world', 'country']
# cols = df_new.columns
# new_cols = [col for col in cols if col not in remove_tag]
# df_new = df_new[new_cols]

# Check for missing tags and print them
for parent_tag, child_tags in tag_pairs:
    missing_tags = [child_tag for child_tag in child_tags if child_tag not in df_new.columns]
    if parent_tag not in df_new.columns or missing_tags:
        print(f"Missing tags for parent '{parent_tag}': {missing_tags}")
# Merge tags and record changes
changes_dict = {}


for parent_tag, child_tags in tag_pairs:
    df_new, changed_indices = update_parent_remove_children(df_new, parent_tag, child_tags)
    changes_dict[parent_tag] = sorted(changed_indices)

# Save the merged dataframe
df_new.to_csv(merged_path, index=False)

# # made by seunghoi start
# cols = df_new.columns
# cols = [f'"{x}"' for x in cols]
# header = '\t'.join(cols)

# data = []
# for i in range(len(df_new)):
#     row = list(df_new.iloc[i])
#     row_tmp = [f'"{x}"' for x in row[:-1]]
#     row_tmp.append(row[-1])
#     string = '\t'.join(row_tmp)
#     data.append(string)

# result = pd.DataFrame(data=data, columns = [header])

# result.to_csv("modified_ann_file.csv", index=False)

# # made by seunghoi end

# # Write the changes to a file
# with open(changes_path, 'w') as file:
#     for parent_tag, indices in changes_dict.items():
#         for index in indices:
#             file.write(f"Parent tag '{parent_tag}' changed in row {index}\n")
