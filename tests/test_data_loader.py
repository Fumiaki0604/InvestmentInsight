from pathlib import Path

from app.data_sources import DataLoader


def test_load_sample_csv() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_path = project_root / "data" / "sample_nav.csv"

    loader = DataLoader(local_path=sample_path)
    df = loader.load_ticker_data("04311047")

    assert not df.empty
    assert df["ticker_id"].iloc[0] == "04311047"
    assert df["date"].notna().all()
