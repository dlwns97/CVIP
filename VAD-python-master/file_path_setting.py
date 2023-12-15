import pandas as pd

path = "/home/seunghoi/clmr/data/magnatagatune/annotations_final_origin.csv"
tag_id_path = "/home/seunghoi/clmr/new_vocal_sex_result.txt"
result_path = "/home/seunghoi/clmr/VAD-python-master"


df = pd.read_csv(path)

with open(tag_id_path, 'r') as file:
    clip_ids = [line.strip() for line in file]

# clip_id와 일치하는 행 찾기 및 mp3_path 추출
mp3_paths = {}
for clip_id in clip_ids:
    # clip_id를 숫자로 변환 (CSV 파일에서 clip_id가 숫자 형태인지 확인 필요)
    clip_id_num = int(clip_id)  
    matched_row = df[df['clip_id'] == clip_id_num]
    
    if not matched_row.empty:
        mp3_path = matched_row.iloc[0]['mp3_path']  # mp3_path 칼럼에서 값 추출
        mp3_paths[clip_id] = mp3_path

# 결과를 JSON 파일로 저장
with open(result_path, 'w') as fp:
    json.dump(mp3_paths, fp)
