# Data

원본 LendingClub 데이터는 저장소에 포함하지 않습니다.

## Raw source

Google Drive에서 확인된 원본 파일:

```text
accepted_2007_to_2018Q4.csv
```

원본 파일은 약 1.67GB 규모이므로 GitHub에 업로드하지 않습니다.

`src/modeling_pipeline.py`는 raw CSV를 기준으로 주요 preprocessing 흐름을 재구성합니다.

## Recovered experiment input

복구된 원본 실험 notebook은 아래 전처리 완료 데이터셋을 사용했습니다.

```text
accepted_df2.csv
```

cleaned notebook을 실행하려면:

```text
data/accepted_df2.csv
```

경로에 해당 파일을 배치하세요.

두 데이터 파일 모두 `.gitignore`에 의해 Git 추적 대상에서 제외됩니다.

데이터 라이선스와 배포 조건은 원본 데이터 제공처의 정책을 따릅니다.
