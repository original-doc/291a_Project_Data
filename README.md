
temp install guide

```

pip install qdrant-client==1.5.4 rank_bm25 numpy 

# install torch by official guide
```

```
# For qdrant, we shall run the backend in a docker

docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/.cache:/qdrant/storage:z" \
    qdrant/qdrant
```