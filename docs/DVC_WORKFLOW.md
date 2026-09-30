# DVC Workflow

This project uses DVC to version the Iris dataset while keeping Git focused on code and metadata.

## Remote
- Default DVC remote: `myremote`
- Local storage path: `C:\Users\HP\dvc-remote-storage`

## Workflow
1. Create or update a dataset file in `data/raw/`.
2. Run `python -m dvc add data/raw/iris_v1.csv` to create a content-addressed `.dvc` pointer.
3. Stage the metadata with `git add data/raw/iris_v1.csv.dvc data/raw/.gitignore`.
4. Commit the change with `git commit -m "data: ..."`.
5. Run `python -m dvc push` so the data object is stored in the remote.
6. To compare versions, use `git log --oneline -- data/raw/iris_v1.csv.dvc` and `python -m dvc diff <commit>`.
7. To restore a historical version, check out the older `.dvc` metafile and run `python -m dvc checkout`.
