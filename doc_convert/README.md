# 문서 → 마크다운 변환

[MarkItDown](https://github.com/microsoft/markitdown)(Microsoft, MIT 라이선스)으로 PDF·워드·엑셀·파워포인트·HTML 등을 마크다운으로 바꾸는 스크립트입니다. 실적 보고서, IR 자료, 강의 자료를 AI에게 넘기기 전에 텍스트로 정리할 때 씁니다.

## 설치

```
pip install -r requirements.txt
```

파이썬 3.10 이상이 필요합니다.

## 사용법

```
# inbox/ 폴더에 파일을 넣고 실행하면 out/ 폴더에 .md가 생깁니다
python convert.py

# 파일 하나, 또는 웹 페이지
python convert.py QCOM_10-Q.pdf
python convert.py https://example.com/article
```

명령어 하나로도 쓸 수 있습니다: `markitdown 파일.pdf -o 결과.md`

## 주의

- 스캔한 PDF(글자가 이미지인 것)는 글자를 읽지 못합니다. 텍스트가 들어 있는 PDF만 됩니다.
- 표가 복잡한 PDF는 줄이 어긋날 수 있으니 숫자는 원본과 대조하세요.
- `out/` 폴더는 깃에 올라가지 않게 해 두었습니다. 개인 자료가 공개 저장소에 올라가지 않도록 `inbox/`도 파일 자체는 제외됩니다.
