"""문서를 마크다운으로 일괄 변환 (Microsoft MarkItDown 사용).

PDF, 워드(.docx), 엑셀(.xlsx), 파워포인트(.pptx), HTML, CSV 등을
마크다운(.md)으로 바꿉니다. 실적 보고서나 IR 자료를 Claude에게
넘기기 전에 텍스트로 정리할 때 쓰면 됩니다.

사용법:
    python convert.py                    # inbox/ 안의 파일 전부 → out/
    python convert.py 보고서.pdf          # 파일 하나 → out/보고서.md
    python convert.py 폴더경로 결과폴더
    python convert.py https://example.com/page   # 웹 페이지도 가능
"""
import sys
from pathlib import Path

from markitdown import MarkItDown

HERE = Path(__file__).parent
EXTS = {".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".html", ".htm",
        ".csv", ".json", ".xml", ".epub", ".zip", ".txt"}


def convert(md: MarkItDown, src: str, out_dir: Path, used: set) -> None:
    result = md.convert(src)
    if src.startswith("http"):
        name = (result.title or "web_page").replace("/", "_")[:80]
    else:
        p = Path(src)
        # 이름이 같은 파일(보고서.pdf, 보고서.xlsx)이 서로 덮어쓰지 않게 확장자를 붙임
        name = p.stem if p.stem not in used else f"{p.stem}_{p.suffix.lstrip('.')}"
    used.add(name)
    out = out_dir / f"{name}.md"
    out.write_text(result.markdown, encoding="utf-8")
    print(f"완료  {src} → {out}  ({len(result.markdown):,}자)")


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "inbox")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = MarkItDown()
    used: set = set()

    if src.startswith("http"):
        targets = [src]
    elif Path(src).is_dir():
        targets = [str(p) for p in sorted(Path(src).iterdir()) if p.suffix.lower() in EXTS]
        if not targets:
            raise SystemExit(f"{src} 안에 변환할 파일이 없습니다.")
    else:
        targets = [src]

    for t in targets:
        try:
            convert(md, t, out_dir, used)
        except Exception as e:  # 한 파일이 실패해도 나머지는 계속
            print(f"실패  {t}: {e}")


if __name__ == "__main__":
    main()
