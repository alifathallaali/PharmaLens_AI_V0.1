from pathlib import Path
import shutil
from enterprise.models import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA = PROJECT_ROOT / "data" / "private"
PRIVATE_DATA.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".parquet"}

def save_dataset(db, uploaded_file, workspace_id: int, user_id: int):
    filename = Path(uploaded_file.name).name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")
    workspace_dir = PRIVATE_DATA / f"workspace_{workspace_id}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    destination = workspace_dir / filename
    with open(destination, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    dataset = Dataset(name=Path(filename).stem, file_name=filename,
                      file_path=str(destination), workspace_id=workspace_id,
                      uploaded_by=user_id)
    db.add(dataset); db.commit(); db.refresh(dataset)
    return dataset
