cd /workspace/project/org-executor
pip install -e . --force

cd -
export TORCH_EXTENSIONS_DIR=$PWD/_build
export ORG_EXECUTOR_WORKSPACE_DIR=$PWD/_build
