# Kubernetes deployment

Local single-node deploy (kind / k3s / minikube). Ollama runs on the **host** GPU, not in-cluster.

## Prereqs
- A local cluster: `kind create cluster` (or k3s/minikube)
- Ollama running on the host, bound so the cluster can reach it:
  `OLLAMA_HOST=0.0.0.0:11434 ollama serve` and `ollama pull nomic-embed-text`
- Images built locally:
  ```bash
  docker build -t grounded-rag-api:local ./api
  docker build -t grounded-rag-web:local ./web
  kind load docker-image grounded-rag-api:local grounded-rag-web:local   # if using kind
  ```

## Deploy
```bash
kubectl apply -k k8s/                       # namespace, config, postgres, api, web
kubectl -n grounded-rag rollout status deploy/postgres
kubectl -n grounded-rag rollout status deploy/api
kubectl apply -f k8s/ingest-job.yaml        # build the index
kubectl -n grounded-rag logs job/ingest -f
```

## Access
```bash
kubectl -n grounded-rag port-forward svc/web 5173:5173 &
kubectl -n grounded-rag port-forward svc/api 8000:8000 &
# open http://localhost:5173
```

## Notes
- **OLLAMA_HOST** in `config.yaml` defaults to `host.docker.internal`. On bare-metal k3s set it
  to the node IP. This is the one value to get right.
- The crystal-battle corpus is mounted read-only via `hostPath`. For a real cluster, bake the data
  into the image or use a proper volume instead.
- The DB password is a plaintext demo Secret — call that out in interviews and mention sealed-secrets
  / external-secrets as the production answer. Showing you *know* it's a shortcut reads better than
  pretending it's production-ready.
