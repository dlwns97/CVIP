import pandas as pd

# CSV 파일 읽기 및 처리
path = '/home/seunghoi/clmr/20231124_after_tag141_v3.csv'
df = pd.read_csv(path)
df = df[df.columns[:-1]]
column_sums = df.sum().to_dict()
sorting_dic = dict(sorted(column_sums.items(), key=lambda item: item[1], reverse=True))
top50_cols = list(sorting_dic.keys())[:51]
new_df = df[top50_cols]
print(top50_cols)

# TSV 파일들의 경로 설정
tsv_files = ['data/magnatagatune/train_gt_mtt.tsv','data/magnatagatune/val_gt_mtt.tsv','data/magnatagatune/test_gt_mtt.tsv']

for file in tsv_files:
    # TSV 파일 읽기
    df_tsv = pd.read_csv(file, sep='\t', header=None, names=['clip_id', 'features'])

    # 새로운 데이터로 업데이트
    df_tsv['features'] = df_tsv['clip_id'].apply(
        lambda x: new_df.loc[new_df['clip_id'] == x, top50_cols[1:]].iloc[0].tolist() if x in new_df['clip_id'].values else []
    )
    # TSV 형식으로 저장
    df_tsv.to_csv(f'{file.split("/")[-1].split(".")[0]}_v4.tsv', sep='\t', index=False, header=False)
