"""/v1/files/{key}?exp&sig: signed GET for FsStorage objects (frames, masks, depth maps, citizen photos).

S3 deployments hand out S3 presigned URLs instead and never hit this route.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from server.app.deps import StorageDep
from server.app.storage import FsStorage, StorageError, guess_content_type

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{key:path}", summary="Download a stored object through a signed URL (FsStorage)")
def get_file(key: str, storage: StorageDep, exp: int = Query(...), sig: str = Query(...)) -> FileResponse:
    if not isinstance(storage, FsStorage):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "storage backend is not fs; use the S3 presigned URL")
    try:
        path = storage.local_path(key)
    except StorageError as e:
        raise HTTPException(422, str(e)) from e
    if not storage.verify("GET", key, exp, sig):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bad or expired signature")
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    return FileResponse(path, media_type=guess_content_type(key),
                        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})
